import torch
import time
import json
import os
from collections import defaultdict

# ── Import the side-channel dict that the model writes to during forward() ──
# This is a module-level global dict, so it survives FSDP wrapping.
try:
    from climate_learn.models.hub.adaptive_side_channel import DIAG
except ImportError:
    try:
        from .adaptive_side_channel import DIAG
    except ImportError:
        DIAG = {}  # fallback: empty dict, metrics will be missing but won't crash


class AdaptiveMonitor:
    """
    Tracks everything you need for the paper + debugging.

    Logged per step:
      - loss
      - difficulty_map statistics (mean, std, min, max, entropy)
      - number of tokens kept vs total
      - batch wall-clock time

    Logged per epoch:
      - all the above, averaged
      - validation metrics (acc, rmse, mse)
      - FLOPs ratio estimate
      - parameter count breakdown

    Saved artifacts:
      - training_log.json (full per-step log)
      - difficulty_histograms/ (per-epoch histograms for paper figures)
    """

    def __init__(self, log_dir=".", world_rank=0, log_interval=50):
        self.log_dir = log_dir
        self.world_rank = world_rank
        self.log_interval = log_interval

        self.step_logs = []
        self.epoch_accumulators = defaultdict(list)
        self._step_start_time = None
        self._logged_params = False

    def start_step(self):
        """Call at the beginning of each training step."""
        if self.world_rank == 0:
            torch.cuda.synchronize()
            self._step_start_time = time.perf_counter()

    def log_step(self, model, loss, batch_idx, epoch, extra_metrics=None):
        """
        Call after loss.backward() each step.

        Extracts adaptive sparsity diagnostics from the DIAG side-channel
        (a module-level dict written to during forward()). This bypasses
        FSDP wrapping entirely.

        Args:
            model: The FSDP-wrapped model (used for param breakdown only).
            loss: Training loss (scalar tensor).
            batch_idx: Current batch index.
            epoch: Current epoch.
            extra_metrics: Optional dict of additional metrics to log.
        """
        if self.world_rank != 0:
            return

        # Wall clock time
        torch.cuda.synchronize()
        step_time = time.perf_counter() - self._step_start_time if self._step_start_time else 0

        entry = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "loss": loss.item(),
            "step_time_sec": round(step_time, 4),
        }

        # ── Difficulty map statistics (from DIAG side-channel) ──
        diff_map = DIAG.get("difficulty_map", None)
        if diff_map is not None:
            entry["difficulty_mean"] = round(diff_map.mean().item(), 4)
            entry["difficulty_std"] = round(diff_map.std().item(), 4)
            entry["difficulty_min"] = round(diff_map.min().item(), 4)
            entry["difficulty_max"] = round(diff_map.max().item(), 4)

            # Entropy of difficulty distribution (how spread out the uncertainty is)
            # High entropy = uniform difficulty, low entropy = bimodal (clear easy/hard split)
            p = diff_map.clamp(1e-7, 1 - 1e-7)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
            entry["difficulty_entropy"] = round(entropy.item(), 4)

            # Fraction of "hard" tokens (difficulty > 0.5)
            entry["frac_hard_tokens"] = round((diff_map > 0.5).float().mean().item(), 4)

        # ── Auxiliary difficulty loss ──
        aux_loss = DIAG.get("aux_loss", None)
        if aux_loss is not None:
            entry["aux_difficulty_loss"] = round(aux_loss.item(), 6)

        # ── Token counts (also try DIAG, fallback to model attributes) ──
        inner = _unwrap_model(model)
        keep_ratio = getattr(inner, 'keep_ratio', None)
        num_patches = getattr(inner, 'num_patches', None)
        if keep_ratio is not None and num_patches is not None:
            K = max(1, int(num_patches * keep_ratio))
            entry["tokens_total"] = num_patches
            entry["tokens_kept"] = K
            entry["tokens_dropped"] = num_patches - K
            entry["keep_ratio"] = keep_ratio

        # ── GPU memory ──
        entry["gpu_mem_allocated_gb"] = round(
            torch.cuda.memory_allocated() / 1e9, 3
        )
        entry["gpu_mem_max_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 3
        )

        # ── Extra metrics ──
        if extra_metrics:
            entry.update(extra_metrics)

        self.step_logs.append(entry)

        # Accumulate for epoch summary
        for k, v in entry.items():
            if isinstance(v, (int, float)):
                self.epoch_accumulators[k].append(v)

        # Console output (every log_interval steps)
        if batch_idx % self.log_interval == 0:
            diff_str = ""
            if "difficulty_mean" in entry:
                diff_str = (
                    f" | diff μ={entry['difficulty_mean']:.3f} "
                    f"σ={entry['difficulty_std']:.3f} "
                    f"hard={entry['frac_hard_tokens']:.1%}"
                )
                if "aux_difficulty_loss" in entry:
                    diff_str += f" aux={entry['aux_difficulty_loss']:.4f}"
            else:
                diff_str = " | diff=N/A (DIAG empty)"

            token_str = ""
            if "tokens_kept" in entry:
                token_str = f" | tokens {entry['tokens_kept']}/{entry['tokens_total']}"

            print(
                f"[E{epoch} B{batch_idx}] "
                f"loss={entry['loss']:.5f} "
                f"time={entry['step_time_sec']:.3f}s"
                f"{diff_str}{token_str}"
                f" | mem={entry['gpu_mem_allocated_gb']:.1f}GB",
                flush=True,
            )

        # Log parameter breakdown once
        if not self._logged_params:
            self._log_param_breakdown(inner)
            self._logged_params = True

    def log_epoch_summary(self, epoch):
        """Print and store epoch-level aggregated metrics."""
        if self.world_rank != 0:
            return

        summary = {"epoch": epoch}
        print(f"\n{'='*70}")
        print(f" Epoch {epoch} Summary (Adaptive Sparsity)")
        print(f"{'='*70}")

        for key, values in self.epoch_accumulators.items():
            if key in ("epoch", "batch_idx"):
                continue
            avg = sum(values) / len(values)
            summary[f"avg_{key}"] = round(avg, 6)
            print(f"  {key:35s} = {avg:.6f}  (n={len(values)})")

        print(f"{'='*70}\n", flush=True)

        # Reset accumulators for next epoch
        self.epoch_accumulators.clear()

        # Save log to disk incrementally
        self._save_log()

        return summary

    def log_validation(self, epoch, val_loss_dict):
        """
        Log validation metrics.

        Args:
            epoch: Current epoch.
            val_loss_dict: Dict from validate_epoch(), e.g.
                {"val/lat_rmse:2m_temperature": 1.23, "val/lat_acc:aggregate": 0.85}
        """
        if self.world_rank != 0:
            return

        entry = {"epoch": epoch, "type": "validation"}
        entry.update({k: round(v, 6) if isinstance(v, float) else v
                      for k, v in val_loss_dict.items()})
        self.step_logs.append(entry)
        self._save_log()

    def _log_param_breakdown(self, model):
        """Print parameter counts by component (once)."""
        components = {
            "cnn_path": "CNN Path (enhanced)",
            "token_selector": "Token Selector",
            "token_restorer": "Token Restorer (mask token)",
            "blocks_early": "ViT Early Blocks (dense)",
            "blocks_middle": "ViT Middle Blocks (sparse)",
            "blocks_late": "ViT Late Blocks (dense)",
            "head": "Decoder Head",
            "token_embeds": "Token Embeddings",
            "var_agg": "Variable Aggregation",
        }

        print(f"\n{'='*70}")
        print(f" Parameter Breakdown")
        print(f"{'='*70}")

        total = 0
        accounted = 0
        for attr, label in components.items():
            module = getattr(model, attr, None)
            if module is not None:
                count = sum(p.numel() for p in module.parameters())
                accounted += count
                print(f"  {label:40s} {count/1e6:8.3f}M  ({count:>12,})")

        total = sum(p.numel() for p in model.parameters())
        other = total - accounted
        print(f"  {'Other (pos_embed, var_embed, etc.)':40s} {other/1e6:8.3f}M  ({other:>12,})")
        print(f"  {'─'*60}")
        print(f"  {'TOTAL':40s} {total/1e6:8.3f}M  ({total:>12,})")

        # FLOPs ratio
        if hasattr(model, 'get_flops_ratio'):
            ratio = model.get_flops_ratio()
            print(f"\n  Attention FLOPs ratio: {ratio:.1%} of dense baseline")
            n_e = getattr(model, 'num_dense_early', '?')
            n_m = getattr(model, 'num_sparse_middle', '?')
            n_l = getattr(model, 'num_dense_late', '?')
            print(f"  Block split: {n_e} dense early → {n_m} sparse middle → {n_l} dense late")
            L = getattr(model, 'num_patches', '?')
            kr = getattr(model, 'keep_ratio', '?')
            print(f"  Tokens: {L} total, keep {kr} = {int(L*kr) if isinstance(L,int) and isinstance(kr,float) else '?'}")

        print(f"{'='*70}\n", flush=True)

    def _save_log(self):
        """Save accumulated logs to JSON."""
        try:
            path = os.path.join(self.log_dir, "adaptive_training_log.json")
            with open(path, 'w') as f:
                json.dump(self.step_logs, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save training log: {e}", flush=True)

    def save_all(self):
        """Final save."""
        self._save_log()
        if self.world_rank == 0:
            print(f"Training log saved to {self.log_dir}/adaptive_training_log.json", flush=True)


def _unwrap_model(model):
    """Unwrap FSDP/DDP/CheckpointWrapper to get the raw model."""
    inner = model
    # Unwrap FSDP
    if hasattr(inner, '_fsdp_wrapped_module'):
        inner = inner._fsdp_wrapped_module
    # Unwrap DDP
    if hasattr(inner, 'module'):
        inner = inner.module
    # Some FSDP versions use _modules
    while hasattr(inner, '_fsdp_wrapped_module'):
        inner = inner._fsdp_wrapped_module
    return inner