import torch
import numpy as np
import glob
import os
import random
import threading
from torch.utils.data import IterableDataset


class SequentialMonthlyDataset(IterableDataset):
    def __init__(self,
                 root_dir,
                 in_vars,
                 out_vars,
                 pred_range=120,
                 subsample=6,
                 transform=None,
                 rank=0,
                 world_size=1,
                 dtype="float32"):
        """
        Reads .npz files sequentially with background prefetching.
        Shards files based on rank to ensure unique data per GPU.

        While the GPU processes samples from file N, a short-lived background
        thread loads file N+1. This overlaps I/O with compute.

        IMPORTANT: For distributed training with FSDP/DDP, this dataset ensures
        all ranks produce the same number of samples to prevent deadlocks by
        limiting each rank to the same number of files.

        Expects npz files with shape (T, 1, H, W) or (T, H, W) for each variable.

        Args:
            root_dir: Directory containing monthly .npz files (e.g., era5/0.25_deg/train)
            in_vars: List of input variable names
            out_vars: List of output variable names
            pred_range: Prediction lead time in timesteps
            subsample: Temporal subsampling factor
            transform: Optional transform function (normalization)
            rank: Current process rank
            world_size: Total number of processes
            dtype: String specifying dtype ("float32", "bfloat16", "float16")
        """
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.history = 1
        self.pred_range = pred_range
        self.subsample = subsample
        self.transform = transform
        self.rank = rank
        self.world_size = world_size

        # Convert string dtype to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map.get(dtype, torch.float32)

        # Get all monthly files
        all_files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if len(all_files) == 0:
            raise ValueError(f"No .npz files found in {root_dir}")

        n_files = len(all_files)

        # Calculate files per rank using floor division to ensure all ranks get the same count
        # This avoids the need for any distributed synchronization
        files_per_rank = n_files // world_size

        if files_per_rank > 0:
            # Enough files to shard across ranks
            start_idx = rank * files_per_rank
            end_idx = start_idx + files_per_rank
            self.files = all_files[start_idx:end_idx]
            print(f"[Rank {rank}] Assigned {len(self.files)}/{n_files} monthly files (indices {start_idx}:{end_idx}).", flush=True)
        else:
            # Fewer files than ranks (e.g., 2 val files with 8 GPUs)
            # All ranks read all files to avoid yielding 0 samples
            self.files = all_files
            print(f"[Rank {rank}] Not enough files to shard ({n_files} files, {world_size} ranks). All ranks read all {n_files} files.", flush=True)

        # Verify first file has required variables (only on rank 0)
        if rank == 0 and len(all_files) > 0:
            self._verify_monthly_file(all_files[0])

    def _verify_monthly_file(self, filepath):
        """Verify a monthly file has required variables."""
        print(f"[Monthly Loader] Verifying variables in {os.path.basename(filepath)}...", flush=True)
        with np.load(filepath) as npz:
            available = list(npz.keys())
            print(f"[Monthly Loader] Available variables: {available}", flush=True)

            all_required = set(self.in_vars + self.out_vars)
            for var in all_required:
                if var not in available:
                    print(f"[Monthly Loader] ERROR: Required var '{var}' NOT FOUND!", flush=True)
                else:
                    shape = npz[var].shape
                    print(f"[Monthly Loader] '{var}': shape={shape}", flush=True)

    def _load_month_to_ram(self, filepath):
        """Load variables from monthly file."""
        try:
            npz = np.load(filepath)
            data = {}

            # Load all required variables
            all_vars = set(self.in_vars + self.out_vars)

            for var in all_vars:
                if var in npz:
                    arr = npz[var]
                    # Handle shape (T, 1, H, W) -> squeeze to (T, H, W)
                    if arr.ndim == 4 and arr.shape[1] == 1:
                        arr = arr.squeeze(1)
                    tensor = torch.from_numpy(arr)
                    if self.dtype != torch.float32:
                        tensor = tensor.to(self.dtype)
                    data[var] = tensor
                else:
                    print(f"[Rank {self.rank}] ERROR: '{var}' not in {os.path.basename(filepath)}")
                    return None

            return data
        except Exception as e:
            print(f"[Rank {self.rank}] Failed to load {filepath}: {e}", flush=True)
            return None

    def _prefetch_file(self, filepath, result_holder):
        """Load a file in a background thread. Stores result in result_holder[0]."""
        result_holder[0] = self._load_month_to_ram(filepath)

    def __iter__(self):
        # Shuffle files for this epoch
        files_to_process = self.files.copy()
        random.shuffle(files_to_process)
        samples_yielded = 0

        # --- Prefetch logic ---
        # While GPU processes samples from file N, load file N+1 in background.
        # This hides most of the I/O latency behind compute.
        prefetch_thread = None
        prefetch_result = [None]  # mutable container for thread to write into

        for file_idx, filepath in enumerate(files_to_process):

            # Wait for prefetch if we started one on the previous iteration
            if prefetch_thread is not None:
                prefetch_thread.join()
                data_dict = prefetch_result[0]
                prefetch_result[0] = None
            else:
                # First file: load synchronously (no prefetch available yet)
                data_dict = self._load_month_to_ram(filepath)

            if data_dict is None:
                continue

            # Start prefetching the NEXT file in background
            if file_idx + 1 < len(files_to_process):
                prefetch_result = [None]
                prefetch_thread = threading.Thread(
                    target=self._prefetch_file,
                    args=(files_to_process[file_idx + 1], prefetch_result),
                    daemon=True,
                )
                prefetch_thread.start()
            else:
                prefetch_thread = None

            # Get time dimension from first variable
            first_var = self.in_vars[0]
            T_total = data_dict[first_var].shape[0]

            # Generate valid time indices
            # Need: t - history >= 0 and t + pred_range - 1 < T_total
            # So: t >= history and t <= T_total - pred_range
            indices = list(range(self.history, T_total - self.pred_range, self.subsample))
            random.shuffle(indices)

            for t in indices:
                try:
                    # Build input tensor: [num_vars * history, H, W]
                    x_list = []
                    for var in self.in_vars:
                        # Slice [t-history : t] -> shape [history, H, W]
                        channel = data_dict[var][t - self.history : t]
                        x_list.append(channel)

                    x = torch.cat(x_list, dim=0)  # [num_vars * history, H, W]

                    # Build output tensor: [num_out_vars, H, W]
                    y_list = []
                    for var in self.out_vars:
                        # Target at t + pred_range - 1
                        target = data_dict[var][t + self.pred_range - 1]  # [H, W]
                        y_list.append(target)

                    y = torch.stack(y_list, dim=0)  # [num_out_vars, H, W]

                    # Apply normalization transform
                    if self.transform:
                        x, y = self.transform(x, y)

                    # Debug: check for NaNs
                    if torch.isnan(x).any():
                        print(f"[Rank {self.rank}] NaN in x at t={t}, file={os.path.basename(filepath)}")
                        continue
                    if torch.isnan(y).any():
                        print(f"[Rank {self.rank}] NaN in y at t={t}, file={os.path.basename(filepath)}")
                        continue

                    yield x, y, self.in_vars, self.out_vars
                    samples_yielded += 1

                except Exception as e:
                    print(f"[Rank {self.rank}] Error at t={t} in {os.path.basename(filepath)}: {e}")
                    continue

            # Free memory
            del data_dict

        # Wait for any remaining prefetch thread
        if prefetch_thread is not None:
            prefetch_thread.join()

        if self.rank == 0:
            print(f"[Monthly Loader] Epoch complete. Yielded {samples_yielded} samples.", flush=True)