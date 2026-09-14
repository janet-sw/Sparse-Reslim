# Standard library
from argparse import ArgumentParser
import os
from pathlib import Path
import torch
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Sequential
from datetime import timedelta
import sys
import time
import yaml

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.models.hub.components.vit_blocks import Block_edm
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock,
)
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
#from climate_learn.dist.profile import *
from utils import seed_everything, init_par_groups


def log_gpu_memory(device, message="", world_rank=None):
    """Log GPU memory usage with optional message and rank."""
    memory_gb_alloc = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024
    memory_gb_reserv = torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024

    if world_rank is not None:
        print(
            f"rank {world_rank} {message} torch.cuda.max_memory_allocated: {memory_gb_alloc:.2f}GB. torch.cuda.memory_reserved: {memory_gb_reserv:.2f}GB reserved",
            flush=True,
        )
    else:
        print(f"{message} torch.cuda.memory_allocated: {memory_gb_alloc:.2f}GB. torch.cuda.memory_reserved: {memory_gb_reserv:.2f}GB reserved.", flush=True)


def get_tensor_parallel_checkpoint_path(base_path, rank, tensor_par_size):
    """Get checkpoint path for tensor parallel models."""
    if tensor_par_size > 1:
        return f"{base_path}_rank_{rank}"
    return base_path


def get_checkpoint_filename(save_path, epoch, world_rank, tensor_par_size):
    """Generate checkpoint filename for given epoch and rank."""
    base_filename = f"{save_path}/interm_epoch_{epoch}.ckpt"
    if tensor_par_size > 1:
        return f"{base_filename}_rank_{world_rank}"
    return base_filename


def load_checkpoint_pretrain(
    model,
    checkpoint_path,
    pretrain_path,
    cp_save_path,
    tensor_par_size=1,
    tensor_par_group=None,
):
    """
    Load model weights from checkpoint or pretrained model.

    This function handles three scenarios:
    1. Resume training from a checkpoint (loads model, optimizer, scheduler states)
    2. Initialize from a pretrained model (loads only model weights)
    3. Initialize model for tensor parallelism when no checkpoint exists

    Args:
        model: PyTorch model to load weights into
        checkpoint_path (str): Path to checkpoint file for resuming training
        pretrain_path (str): Path to pretrained model weights
        cp_save_path (str): Directory where checkpoints will be saved
        tensor_par_size (int): Size of tensor parallelism (default: 1)
        tensor_par_group: Process group for tensor parallelism (default: None)

    Returns:
        tuple: (model, start_epoch) where start_epoch is the epoch to resume from
    """
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    # load model checkpoint
    if checkpoint_path is not None and world_rank < tensor_par_size:

        checkpoint_path = get_tensor_parallel_checkpoint_path(
            checkpoint_path, world_rank, tensor_par_size
        )

        if os.path.exists(checkpoint_path):

            print(
                "world_rank",
                world_rank,
                "model resume from checkpoint",
                checkpoint_path,
                " Checkpoint path found.",
                flush=True,
            )

            map_location = "cpu"

            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint["model_state_dict"])

            del checkpoint
        else:
            print(
                "resume from checkpoint was set to True. "
                "But the checkpoint path does not exist.",
                flush=True,
            )
            sys.exit("checkpoint path does not exist")

    # load pretrained model
    if pretrain_path is not None and world_rank < tensor_par_size:
        if tensor_par_size > 1:
            pretrain_path = pretrain_path + "_" + "rank" + "_" + str(world_rank)

        if os.path.exists(pretrain_path):
            print(
                "world_rank",
                world_rank,
                "load pretrained model",
                pretrain_path,
                " Pretrain path found.",
                flush=True,
            )
            _load_pretrained_weights(model, pretrain_path, local_rank, world_rank)
        else:
            print(
                "resume from pretrained model was set to True. "
                "But the pretrained model path does not exist.",
                flush=True,
            )
            sys.exit("pretrain path does not exist")

    # initialize weights for tensor parallelism when training from scratch
    if pretrain_path is None and checkpoint_path is None and tensor_par_size > 1:
        if world_rank == 0:
            isExist = os.path.exists(cp_save_path)

            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(cp_save_path)
                print("The new checkpoint saving directory is created!")

            # Save initial model weights and distribute to all GPUs in the tensor
            # parallel group to synchronize model weights that do not belong to the
            # training block

            init_model_dict = {
                k: v
                for k, v in model.state_dict().items()
                if ("attn" not in k and "mlp" not in k and "var_agg" not in k)
            }

            print(
                "training from scratch and tensor_par_size>1. rank",
                world_rank,
                "init_model_dict.keys()",
                init_model_dict.keys(),
                flush=True,
            )

            torch.save(
                init_model_dict,
                cp_save_path + "/initial_" + str(dist.get_rank()) + ".pth",
            )

            del init_model_dict

        dist.barrier(device_ids=[local_rank])

        if world_rank != 0 and world_rank < tensor_par_size:

            # Load initial model weights and synchronize model weights that are not
            # in the training block among sequence parallel GPUs
            print("training from scratch. rank", world_rank, flush=True)

            map_location = "cpu"
            model.load_state_dict(
                torch.load(
                    cp_save_path + "/initial_" + str(0) + ".pth",
                    map_location=map_location,
                ),
                strict=False,
            )


def _load_pretrained_weights(model, pretrain_path, device, world_rank):
    map_location = "cpu"
    checkpoint = torch.load(pretrain_path, map_location=map_location)

    print("Loading pre-trained checkpoint from: %s" % pretrain_path)
    pretrain_model = checkpoint["model_state_dict"]

    del checkpoint

    state_dict = model.state_dict()

    if torch.distributed.get_rank() == 0:
        for k in list(pretrain_model.keys()):
            print(
                "Pretrained model before deletion. Name ",
                k,
                "shape",
                pretrain_model[k].shape,
                flush=True,
            )

    for k in list(
        pretrain_model.keys()
    ):  # in pre-train model weights, but not fine-tuning model
        if k not in state_dict.keys():
            print(f"Removing key {k} from pretrained checkpoint: no exist")
            del pretrain_model[k]
        elif (
            pretrain_model[k].shape != state_dict[k].shape
        ):  # if pre-train and fine-tune model weights dimension doesn't match
            if k == "pos_embed":
                print("interpolate positional embedding", flush=True)
                interpolate_pos_embed(model, pretrain_model, new_size=model.compressed_img_size)
            else:
                print(
                    f"Removing key {k} from pretrained checkpoint: no matching shape",
                    pretrain_model[k].shape,
                    state_dict[k].shape,
                )
                del pretrain_model[k]

    # load pre-trained model
    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model


"""
Setup sequence, data, tensor model, and sequence_plus_data parallel groups
"""

def clip_replace_constant(y, yhat, out_variables,data_key):
    """
    y:    ground truth [B, C, H, W]
    yhat: model output [B, C, H, W], requires grad
    """

    output = yhat.clone()

    # ---- Enforce non-negative precipitation ----
    if "total_precipitation_24hr" in out_variables:
        prcp_index = out_variables.index("total_precipitation_24hr")

        if dist.get_rank() == 0:
            print("prcp_index", prcp_index, flush=True)

        output[:, prcp_index] = torch.clamp(yhat[:, prcp_index], min=0.0)

    # ---- Replace constants with ground truth (no grad) ----
    for i, var in enumerate(out_variables):
        if var in CONSTANTS:
            if dist.get_rank() == 0:
                print("reaching constant replacement", flush=True)

            output[:, i] = y[:, i].detach()


    return output


def unpatchify(patchified_pixel_values,data, patch_size):
    original_x, original_y = data.shape[2], data.shape[3]

    num_patches_x = original_x // patch_size
    num_patches_y = original_y // patch_size

    batch_size = patchified_pixel_values.shape[0]
    num_channels = data.shape[1]
    patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_x, num_patches_y, patch_size, patch_size, num_channels)
    patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
    pixel_values = patchified_pixel_values.reshape(batch_size, num_channels, num_patches_x*patch_size, num_patches_y*patch_size)
    return pixel_values

import torch
import torch.nn as nn

class EDM_Scheduler(nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, weight_mode=None,max_weight=100):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.weight_mode = weight_mode
        self.max_weight = max_weight

        if dist.get_rank()==0:
            print("self.weight_mode",self.weight_mode,flush=True)

    def forward(self, images, labels=None, augment_pipe=None):
        #[B, T, Variables, H, W]
        rnd_normal = torch.randn(images.shape[0],1, 1, 1, 1, device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        if self.weight_mode == "clamp":
            weight = torch.clamp(weight,max=self.max_weight)

        n = torch.randn_like(images).mul_(sigma)

        return n, sigma, weight



def save_prediction_png(
    D_yn,
    y,
    current_noisy,
    conditions,
    epoch,
    batch_idx,
    output_dir,
):
    """
    Save first batch, first output variable as PNG.
    """


    # Pick first sample & first variable
    pred = D_yn[0, 0]   # [H, W]
    gt   = y[0, 0]      # [H, W]

    current_noisy = current_noisy[0,0,0]  #[H,W]
    conditions_1 =conditions[0,0,3]
    conditions_2 = conditions[0,1,3]


    if dist.get_rank()==0:
        print("current_noisy.shape",current_noisy.shape,"torch.min(conditions_1)",torch.min(conditions_1),"torch.max(conditions_1)",torch.max(conditions_1),"torch.mean(conditions_1)",torch.mean(conditions_1),"torch.std(conditions_1)",torch.std(conditions_1),"torch.min(conditions_2)",torch.min(conditions_2),"torch.max(conditions_2)",torch.max(conditions_2),"torch.min(pred)",torch.min(pred),"torch.max(pred)",torch.max(pred),"torch.min(gt)",torch.min(gt),"torch.max(gt)",torch.max(gt),flush=True)


    # Detach + move to CPU
    pred = pred.detach().cpu()
    gt   = gt.detach().cpu()
    current_noisy =current_noisy.detach().cpu()
    conditions_1 = conditions_1.detach().cpu()
    conditions_2 = conditions_2.detach().cpu()

    # Normalize to [0,1] for visualization
    def normalize(x):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x

    pred = normalize(pred)
    gt   = normalize(gt)
    current_noisy = normalize(current_noisy)
    conditions_1 = normalize(conditions_1)
    conditions_2 = normalize(conditions_2)

    # Convert to PIL images
    from torchvision.transforms.functional import to_pil_image
    pred_img = to_pil_image(pred.unsqueeze(0))  # [1,H,W]
    gt_img   = to_pil_image(gt.unsqueeze(0))
    current_noisy = to_pil_image(current_noisy.unsqueeze(0))
    conditions_1 = to_pil_image(conditions_1.unsqueeze(0))
    conditions_2 = to_pil_image(conditions_2.unsqueeze(0))


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"epoch_{epoch:03d}_batch_{batch_idx:03d}"
    pred_path = f"{prefix}_pred.png"
    gt_path = f"{prefix}_gt.png"
    current_noisy_path = f"{prefix}_noisy_input.png"
    conditions1_path = f"{prefix}_conditions1.png"
    conditions2_path = f"{prefix}_conditions2.png"





    pred_img.save(pred_path)
    gt_img.save(gt_path)
    current_noisy.save(current_noisy_path)
    conditions_1.save(conditions1_path)
    conditions_2.save(conditions2_path)


    if dist.get_rank()==0:
        print(f"[PNG] Saved prediction → {pred_path}")
        print(f"[PNG] Saved ground truth → {gt_path}")



def training_step(
    batch, batch_idx, net, device: int, var_weights, train_loss_metric,
    batch_size, patch_size, noise_scheduler, precision_dt, epoch, data_key,
    preview_dir=None,
) -> torch.Tensor:
    conditions, y, in_variables, out_variables = batch
    conditions = conditions.to(device, non_blocking=True)
    y = y.to(device,non_blocking=True)


    y = torch.unsqueeze(y,dim = 1)

    n, sigma, diffusion_weights = noise_scheduler(y)


    if precision_dt==torch.bfloat16:
        with torch.autocast(device_type='cuda', dtype=precision_dt):

            D_yn = net(y+n, in_variables,out_variables, sigma, conditions)

            y = y.squeeze(dim=1)
            diffusion_weights = diffusion_weights.squeeze(dim=4)


            if "total_precipitation_24hr" in out_variables:
                D_yn = clip_replace_constant(y,D_yn,out_variables,data_key)


            losses = train_loss_metric(
                D_yn, y, var_names=out_variables, var_weights=var_weights,diffusion_weights=diffusion_weights,
            )

    else:

        D_yn = net(y+n, in_variables, out_variables, sigma, conditions)



        y = y.squeeze(dim=1)
        diffusion_weights = diffusion_weights.squeeze(dim=4)


        if "total_precipitation_24hr" in out_variables:
            D_yn = clip_replace_constant(y,D_yn,out_variables,data_key)



        losses = train_loss_metric(
            D_yn, y, var_names=out_variables, var_weights=var_weights,diffusion_weights=diffusion_weights,
        )


    if preview_dir and dist.get_rank() == 0 and batch_idx == 0:
        save_prediction_png(
            D_yn,
            y,
            y.unsqueeze(dim=1) + n,
            conditions,
            epoch,
            batch_idx,
            preview_dir,
        )



    loss_name = getattr(train_loss_metric, "name", "loss")
    if losses.dim() == 0:  # aggregate loss only
        loss = losses
    else:  # per channel + aggregate
        loss = losses[-1]


    return loss



def validation_step(
    batch, batch_idx: int, net, device: int, val_loss_metrics, val_target_transforms
) -> torch.Tensor:

    return evaluate_func(
        batch, "val", net, device, val_loss_metrics, val_target_transforms
    )


def evaluate_func(batch, stage: str, net, device: int, loss_metrics, target_transforms):

    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)

    yhat = net.forward(x, in_variables, out_variables)
    yhat = clip_replace_constant(y, yhat, out_variables)

    if stage == "val":
        loss_fns = loss_metrics
        transforms = target_transforms
    elif stage == "test":
        loss_fns = loss_metrics
        transforms = target_transforms
    else:
        raise RuntimeError("Invalid evaluation stage")
    loss_dict = {}
    for i, lf in enumerate(loss_fns):

        if transforms is not None and transforms[i] is not None:
            yhat_ = transforms[i](yhat)
            y_ = transforms[i](y)

        if y_.size(dim=2) != yhat_.size(dim=2) or y_.size(dim=3) != yhat_.size(dim=3):
            losses = lf(yhat_, y_[:, :, 0 : yhat_.size(dim=2), 0 : yhat_.size(dim=3)])
        else:
            losses = lf(yhat_, y_)

        loss_name = getattr(lf, "name", f"loss_{i}")
        if losses.dim() == 0:  # aggregate loss
            loss_dict[f"{stage}/{loss_name}:agggregate"] = losses
        else:  # per channel + aggregate
            for var_name, loss in zip(out_variables, losses):
                name = f"{stage}/{loss_name}:{var_name}"
                loss_dict[name] = loss
            loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
    return loss_dict



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    cp_save_path,
    world_rank,
    local_rank,
    tensor_par_size,
    device,
):
    """
    Save model checkpoint to disk.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch (int): Current epoch number
        cp_save_path (str): Directory path for saving checkpoints
        world_rank (int): Current process rank
        local_rank (int): Local rank for the current process
        tensor_par_size (int): Size of tensor parallelism
        device: Current device
    """
    # Create checkpoint directory if needed (only on rank 0)
    if world_rank == 0:
        if not os.path.exists(cp_save_path):
            os.makedirs(cp_save_path)
            print(f"Created checkpoint directory: {cp_save_path}", flush=True)

    # Log memory before saving
    if world_rank == 0:
        log_gpu_memory(device, "Before torch.save", world_rank)

    # Get model, optimizer, and scheduler states
    model_states = model.state_dict()
    optimizer_states = optimizer.state_dict()
    scheduler_states = scheduler.state_dict()

    # Save checkpoint only for ranks that are part of tensor parallelism
    if world_rank < tensor_par_size:
        file_name = get_checkpoint_filename(
            cp_save_path, epoch, world_rank, tensor_par_size
        )

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_states,
            "optimizer_state_dict": optimizer_states,
            "scheduler_state_dict": scheduler_states,
        }

        torch.save(checkpoint_dict, file_name)

        if world_rank == 0:
            print(f"Saved checkpoint to: {file_name}", flush=True)

    # Log memory after saving
    log_gpu_memory(device, "After torch.save", world_rank)

    # Synchronize all processes
    dist.barrier(device_ids=[local_rank])

    # Clean up to free memory
    del model_states
    del optimizer_states
    del scheduler_states


def main(device):
    """
    Main training function for intermediate downscaling model.

    This function coordinates the entire training pipeline:
    1. Sets up distributed training environment
    2. Loads configuration from YAML file
    3. Initializes parallel process groups
    4. Creates data loaders for multiple datasets
    5. Builds and distributes the model across GPUs
    6. Runs training loop with checkpointing

    Args:
        device: Initial device (will be overridden based on local rank)
    """

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    print(
        "world_size",
        world_size,
        "world_rank",
        world_rank,
        "local_rank",
        local_rank,
        flush=True,
    )

    config_path = sys.argv[1]

    if world_rank == 0:
        print("config_path", config_path, flush=True)

    conf = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    max_epochs = conf["trainer"]["max_epochs"]
    checkpoint_path = conf["trainer"]["checkpoint"]
    batch_size = conf["trainer"]["batch_size"]
    num_workers = conf["trainer"]["num_workers"]
    buffer_size = conf["trainer"]["buffer_size"]
    data_type = conf["trainer"]["data_type"]
    gpu_type = conf["trainer"]["gpu_type"]
    train_loss_str = conf["trainer"]["train_loss"]
    pretrain_path = conf["trainer"]["pretrain"]

    fsdp_size = conf["parallelism"]["fsdp"]
    simple_ddp_size = conf["parallelism"]["simple_ddp"]
    tensor_par_size = conf["parallelism"]["tensor_par"]

    try:
        do_tiling = conf["tiling"]["do_tiling"]
        if do_tiling:
            div = conf["tiling"]["div"]
            overlap = conf["tiling"]["overlap"]
        else:
            div = 1
            overlap = 0
    except Exception:
        do_tiling = False
        div = 1
        overlap = 0

    compress_ratio = conf["compression"]["compress_ratio"]

    sigma_data = conf["EDM_hyper"]["sigma_data"]
    weight_mode = conf["EDM_hyper"]["weight_mode"]

    try:
        use_residual_path = conf["EDM_hyper"]["use_residual_path"]
    except Exception:
        use_residual_path = True

    try:
        compress_cond_only = conf["compression"]["compress_cond_only"]
    except Exception:
        compress_cond_only = False

    try:
        learned_compression = conf["compression"]["learned_compression"]
    except Exception:
        learned_compression = False

    data_dir = conf["data"]["data_dir"]
    src = conf["data"]["src"]
    pred_range = conf["data"]["pred_range"]
    history = conf["data"]["history"]
    window = conf["data"]["window"]
    preset = conf["model"]["preset"]
    var_weights = conf["data"]["var_weights"]
    dict_in_variables = conf["data"]["dict_in_variables"]
    dict_out_variables = conf["data"]["dict_out_variables"]
    default_vars = conf["data"]["default_vars"]
    spatial_resolution = conf["data"]["spatial_resolution"]

    lr = float(conf["model"]["lr"])
    beta_1 = float(conf["model"]["beta_1"])
    beta_2 = float(conf["model"]["beta_2"])
    weight_decay = float(conf["model"]["weight_decay"])
    warmup_epochs = conf["model"]["warmup_epochs"]
    warmup_start_lr = float(conf["model"]["warmup_start_lr"])
    eta_min = float(conf["model"]["eta_min"])

    patch_size = conf["model"]["patch_size"]
    embed_dim = conf["model"]["embed_dim"]
    depth = conf["model"]["depth"]
    decoder_depth = conf["model"]["decoder_depth"]
    num_heads = conf["model"]["num_heads"]
    mlp_ratio = conf["model"]["mlp_ratio"]
    drop_path = conf["model"]["drop_path"]
    drop_rate = conf["model"]["drop_rate"]
    keep_ratio = conf["model"].get("keep_ratio", 0.25)
    num_dense_early = conf["model"].get("num_dense_early", 2)
    num_sparse_middle = conf["model"].get("num_sparse_middle")

    data_par_size = fsdp_size * simple_ddp_size

    if world_rank == 0:
        print("\n" + "=" * 80)
        print("Training Configuration Summary")
        print("=" * 80)
        print(f"Model: {preset}, Parameters: {embed_dim}d {depth}L {num_heads}H")
        print(f"Training: {max_epochs} epochs, batch_size={batch_size}, lr={lr}")
        print(f"Data type: {data_type}, Loss: {train_loss_str}")
        print(f"Checkpoint: {checkpoint_path if checkpoint_path else 'None'}")
        print(f"Pretrain: {pretrain_path if pretrain_path else 'None'}")
        print("=" * 80 + "\n", flush=True)
        print(
            "data_par_size",
            data_par_size,
            "fsdp_size",
            fsdp_size,
            "simple_ddp_size",
            simple_ddp_size,
            "tensor_par_size",
            tensor_par_size,
            "division",
            div,
            "overlap",
            overlap,
            flush=True,
        )

    # initialize parallelism groups
    (
        seq_par_group,
        data_par_group,
        tensor_par_group,
        data_seq_ort_group,
        fsdp_group,
        simple_ddp_group,
    ) = init_par_groups(
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        seq_par_size=1, #seq_par not implemented
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
        num_heads=num_heads,
    )

    noise_scheduler = EDM_Scheduler(sigma_data=sigma_data,weight_mode=weight_mode).to(device)

    # choose Attention kernel
    if gpu_type == "amd":
        if data_type == "bfloat16":
            FusedAttn_option = FusedAttn.CK
        else:
            FusedAttn_option = FusedAttn.DEFAULT
    else:
        FusedAttn_option = FusedAttn.DEFAULT

    model_kwargs = {
        "default_vars": default_vars,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "decoder_depth": decoder_depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "drop_path": drop_path,
        "drop_rate": drop_rate,
        "tensor_par_size": tensor_par_size,
        "tensor_par_group": tensor_par_group,
        "FusedAttn_option": FusedAttn_option,
        "compress_ratio": compress_ratio,
        "sigma_data": sigma_data,
        "use_residual_path": use_residual_path,
        "compress_cond_only": compress_cond_only,
        "learned_compression": learned_compression,
        "keep_ratio": keep_ratio,
        "num_dense_early": num_dense_early,
        "num_sparse_middle": num_sparse_middle,
    }

    if world_rank == 0:
        print("model_kwargs", model_kwargs, flush=True)

    if preset != "edm":
        print("Only supports edm training.", flush=True)
        sys.exit("Not edm architecture")

    # if both checkpoint and pretrain are available, use checkpoint
    if checkpoint_path is not None and pretrain_path is not None:
        pretrain_path = None

    model = None

    first_time_bool = True

    interval_epochs = 1

    epoch_start = 0

    cp_save_path = conf["trainer"].get("cp_save_path", "outputs/edm")
    preview_dir = conf["trainer"].get("preview_dir")

    if data_type == "bfloat16":
        scaler = ShardedGradScaler(init_scale=8192, growth_interval=100)
        min_scale = 128
        if world_rank == 0:
            print("initialize ShardedGradScaler for bfloat16", flush=True)

    while (epoch_start + interval_epochs) < max_epochs:

        for data_key in data_dir.keys():
            # Set up data

            in_vars = dict_in_variables[data_key]
            out_vars = dict_out_variables[data_key]

            if world_rank == 0:
                print("***************************", flush=True)
                print("data_key is ", data_key, flush=True)
                print("in_vars", in_vars, flush=True)
                print("out_vars", out_vars, flush=True)
                print("default_vars", default_vars, flush=True)
                log_gpu_memory(device, "before data_module")

            # load data module
            data_module = cl.data.IterDataModule(
                preset,
                data_dir[data_key],
                data_dir[data_key],
                in_vars,
                out_vars=out_vars,
                data_par_size=data_par_size,
                data_par_group=data_par_group,
                subsample=1,
                batch_size=batch_size,
                buffer_size=buffer_size,
                num_workers=num_workers,
                div=div,
                overlap=overlap,
                src=src[data_key],
                pred_range=pred_range[data_key],
                history=history[data_key],
                window=window[data_key],
                pin_memory=True,
                persistent_workers=True
            ).to(device)

            data_module.setup()

            if do_tiling:
                in_lat, in_lon, out_lat, out_lon = data_module.get_lat_lon_dims()
                # yout = len(out_lat) // div
                yinp = in_lat//div + overlap
                if yinp % patch_size != 0:
                    if world_rank == 0:
                        print(f"Tile height: {yinp}, patch_size {patch_size}",flush=True)
                        print(
                            "Overlap must be adjusted to accommodate patch_size of the "
                            "Transformer. Need to decrease the overlap by ",
                            (yinp % patch_size),flush=True
                        )
                        sys.exit(
                            "Please increase the overlap accordingly to the instructions "
                            "in the print message"
                        )

            if world_rank == 0:
                log_gpu_memory(device, "after data_module")

            if first_time_bool:
                # Set up deep learning model
                (
                    model,
                    train_loss,
                    val_losses,
                    test_losses,
                    train_transform,
                    val_transforms,
                    test_transforms,
                ) = cl.load_edm_module(
                    device,
                    model=model,
                    data_module=data_module,
                    architecture=preset,
                    train_loss=train_loss_str,
                    model_kwargs=model_kwargs,
                )

                # Model and loss functions created successfully

                model = model.to(device)


                if torch.distributed.get_rank() == 0:
                    print("model is ",model,flush=True)
                    print("Loading model weights...", flush=True)
                    # Print only model summary instead of all parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    print(
                        f"Total model parameters: {total_params / 1e6:.2f}M", flush=True
                    )

                # load from checkpoint for continued training , or from pretrained model weights
                load_checkpoint_pretrain(
                    model,
                    checkpoint_path,
                    pretrain_path,
                    cp_save_path,
                    tensor_par_size=tensor_par_size,
                    tensor_par_group=tensor_par_group,
                )

                # Model weights loaded, no need to print all parameters again

                if checkpoint_path is None:
                    seed_everything(0)

                # set up layer wrapping
                if preset == "vit" or preset == "res_slimvit" or preset == "ddpm" or preset == "edm":

                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={
                            Block_edm,
                            Sequential,  # < ---- Your Transformer layer class
                        },
                    )

                    check_fn = lambda submodule: isinstance(
                        submodule, Block_edm
                    ) or isinstance(submodule, Sequential)

                if data_type == "float32":
                    precision_dt = torch.float32
                elif data_type == "bfloat16":
                    precision_dt = torch.bfloat16
                else:
                    raise RuntimeError("Data type not supported")

                # floating point policy
                bfloatPolicy = MixedPrecision(
                    param_dtype=precision_dt,
                    # Gradient communication precision.
                    reduce_dtype=precision_dt,
                    # Buffer precision.
                    buffer_dtype=precision_dt,
                )

                # hybrid sharded FSDP
                if fsdp_size > 1 and simple_ddp_size > 1:

                    print("enter hybrid FSDP", flush=True)
                    model = FSDP(
                        model,
                        device_id=local_rank,
                        process_group=(fsdp_group, simple_ddp_group),
                        sync_module_states=True,
                        sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=bfloatPolicy,
                        forward_prefetch=True,
                        limit_all_gathers=False,
                    )
                # fully sharded FSDP
                elif fsdp_size > 1 and simple_ddp_size == 1:
                    print("enter fully sharded FSDP", flush=True)
                    model = FSDP(
                        model,
                        device_id=local_rank,
                        process_group=fsdp_group,
                        sync_module_states=True,
                        sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=bfloatPolicy,
                        forward_prefetch=True,
                        limit_all_gathers=False,
                    )
                else:
                    # no shard only
                    print("enter NO SHARD only,", flush=True)
                    model = FSDP(
                        model,
                        device_id=local_rank,
                        process_group=simple_ddp_group,
                        sync_module_states=True,
                        sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=bfloatPolicy,
                        forward_prefetch=True,
                        limit_all_gathers=False,
                    )

            # Update spatial resolution, image size, and number of variables to model
            # based on datasets
            in_shape, _ = data_module.get_data_dims()

            if preset != "edm":
                _, in_height, in_width = in_shape[1:]
            else:
                _,_,in_height,in_width = in_shape[1:]
                print("in_height",in_height,"in_width",in_width,flush=True)


            with FSDP.summon_full_params(model):
                model.data_config(
                    spatial_resolution[data_key],
                    (in_height, in_width),
                    len(in_vars),
                    len(out_vars),
                )


            if dist.get_rank()==0:
                print("after data_config",flush=True)

            if first_time_bool:
                # activation checkpointing
                apply_activation_checkpointing(
                    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
                )

                #torch.compile for speedup and reduced memory used
                #model = torch.compile(model, mode="max-autotune")


                # load optimzier and scheduler

                optimizer = cl.load_optimizer(
                    model,
                    "adamw",
                    {"lr": lr, "weight_decay": weight_decay, "betas": (beta_1, beta_2)},
                )

                scheduler = cl.load_lr_scheduler(
                    "linear-warmup-cosine-annealing",
                    optimizer,
                    {
                        "warmup_epochs": warmup_epochs,
                        "max_epochs": max_epochs,
                        "warmup_start_lr": warmup_start_lr,
                        "eta_min": eta_min,
                    },
                )

                if checkpoint_path is not None:

                    print(
                        "optimizer resume from checkpoint",
                        checkpoint_path,
                        " Checkpoint path found.",
                        flush=True,
                    )
                    src_rank = world_rank - tensor_par_size * dist.get_rank(
                        group=data_seq_ort_group
                    )
                    map_location = "cpu"
                    checkpoint_path = get_tensor_parallel_checkpoint_path(
                        checkpoint_path, src_rank, tensor_par_size
                    )

                    checkpoint = torch.load(checkpoint_path, map_location=map_location)
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    epoch_start = checkpoint["epoch"] + 1
                    del checkpoint

            # get latitude and longitude
            lat, lon = data_module.get_lat_lon()

            # get train data loader
            train_dataloader = data_module.train_dataloader()

            # get validation data loader
            val_dataloader = data_module.val_dataloader()

            # perform training

            epoch_end = epoch_start + interval_epochs
            epoch_end = epoch_end if epoch_end < max_epochs else max_epochs

            for epoch in range(epoch_start, epoch_end):

                # tell the model that we are in train mode. Matters because we have the dropout
                model.train()
                loss = 0.0
                epoch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                #if world_rank == 0:
                print("epoch ", epoch, flush=True)

                for batch_idx, batch in enumerate(train_dataloader):

                    if world_rank == 0:
                        torch.cuda.synchronize(device=device)
                        tic1 = time.perf_counter()

                    # torch.Size([64, 20, 32, 64]), torch.Size([64, 1, 128, 256])
                    loss = training_step(
                        batch, batch_idx, model, device, var_weights, train_loss,
                        batch_size, patch_size, noise_scheduler, precision_dt,
                        epoch, data_key, preview_dir,
                    )

                    epoch_loss += loss.detach()

                    if world_rank < tensor_par_size:
                        print(
                            "epoch: ",
                            epoch,
                            "batch_idx",
                            batch_idx,
                            "world_rank",
                            world_rank,
                            " loss ",
                            loss.detach(),
                            flush=True,
                        )

                    optimizer.zero_grad(set_to_none=True)

                    if data_type == "float32":
                        loss.backward()



                        grad_norm = torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_(
                            model,
                            max_norm=5.0,
                        )
                        if grad_norm >4.99:
                            print("gradient clipped",flush=True)


                        optimizer.step()
                    else:

                        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                        scaler.scale(loss).backward()

                        # scaler.step() first unscales gradients of the optimizer's params.
                        # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                        # otherwise, optimizer.step() is skipped.
                        scaler.step(optimizer)
                        # Updates the scale for next iteration.
                        scaler.update()
                        if scaler._scale < min_scale:
                            scaler._scale = torch.tensor(min_scale).to(scaler._scale)

                    if world_rank == 0:
                        torch.cuda.synchronize(device=device)
                        tic4 = time.perf_counter()
                        print(
                            f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",
                            flush=True,
                        )

                if world_rank == 0:
                    log_gpu_memory(
                        device,
                        f"batch_idx {batch_idx} get_lr {scheduler.get_last_lr()} after optimizer step. Epoch: {epoch} with epoch_loss {epoch_loss}",world_rank,
                    )


                scheduler.step()

                # Save checkpoint at the end of epoch
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    cp_save_path,
                    world_rank,
                    local_rank,
                    tensor_par_size,
                    device,
                )

            epoch_start = epoch_end

            if first_time_bool:
                first_time_bool = False


if __name__ == "__main__":

    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["RANK"] = os.environ["SLURM_PROCID"]

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=7200000),
        rank=world_rank,
        world_size=world_size,
    )

    print("Using dist.init_process_group. world_size ", world_size, flush=True)

    main(device)

    dist.destroy_process_group()
