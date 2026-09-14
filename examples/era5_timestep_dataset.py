"""
ERA5TimestepDataset — drop-in MapDataset replacement for SequentialMonthlyDataset.

Uses pre-converted per-timestep .npy files for instant random access.
Works with standard PyTorch DataLoader(num_workers=4, shuffle=True).

IMPORTANT: On first run, builds a sample index and caches it to
`{data_dir}/sample_index.json`. Subsequent runs (and worker processes)
load this index instantly instead of scanning 341k files on Lustre.
"""

import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


def build_sample_index(data_dir, pred_range, subsample, history):
    """Scan directory and build list of valid (input, target) pairs.

    Returns list of (input_filenames, target_filename) tuples.
    This is expensive on Lustre (~minutes for 341k files), so we cache it.
    """
    pattern = re.compile(r"^(\d{4}_\d{2})_(\d{3})\.npy$")

    groups = defaultdict(dict)  # prefix -> {step: filename}
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if m:
            prefix = m.group(1)
            step = int(m.group(2))
            groups[prefix][step] = fname

    all_samples = []
    for prefix in sorted(groups.keys()):
        step_map = groups[prefix]
        max_step = max(step_map.keys())

        for t in range(history, max_step - pred_range + 1, subsample):
            input_steps = list(range(t - history + 1, t + 1))
            target_step = t + pred_range

            if not all(s in step_map for s in input_steps):
                continue
            if target_step not in step_map:
                continue

            input_fnames = [step_map[s] for s in input_steps]
            target_fname = step_map[target_step]
            all_samples.append((input_fnames, target_fname))

    return all_samples


def get_or_build_index(data_dir, pred_range, subsample, history, rank):
    """Load cached sample index or build and cache it.

    Cache file: {data_dir}/sample_index_pr{pred_range}_ss{subsample}_h{history}.json
    """
    cache_name = f"sample_index_pr{pred_range}_ss{subsample}_h{history}.json"
    cache_path = os.path.join(data_dir, cache_name)

    if os.path.exists(cache_path):
        if rank == 0:
            print(f"[ERA5TimestepDataset] Loading cached index: {cache_path}", flush=True)
        with open(cache_path) as f:
            return json.load(f)

    # Build index (only rank 0 should do this, but all ranks might race here)
    if rank == 0:
        print(f"[ERA5TimestepDataset] Building sample index for {data_dir}...", flush=True)

    samples = build_sample_index(data_dir, pred_range, subsample, history)

    # Convert tuples to lists for JSON serialization
    samples_json = [[inp, tgt] for inp, tgt in samples]

    # Try to save cache (might fail if read-only, that's ok)
    try:
        if rank == 0:
            with open(cache_path, "w") as f:
                json.dump(samples_json, f)
            print(f"[ERA5TimestepDataset] Cached index: {cache_path} "
                  f"({len(samples)} samples)", flush=True)
    except (OSError, PermissionError):
        if rank == 0:
            print(f"[ERA5TimestepDataset] Could not cache index (read-only?), "
                  f"will rebuild next time", flush=True)

    return samples_json


class ERA5TimestepDataset(Dataset):
    """Map-style dataset over per-timestep .npy files.

    File naming convention: {year}_{month}_{timestep:03d}.npy
    Each file: (C, H, W) float16 array with all variables.

    Uses a cached sample index to avoid scanning 341k files on every init.
    """

    def __init__(self, data_dir, var_names_path, in_vars, out_vars,
                 pred_range=120, subsample=6, history=1,
                 transform=None, rank=0, world_size=1, dtype="bfloat16"):
        self.data_dir = data_dir
        self.pred_range = pred_range
        self.subsample = subsample
        self.history = history
        self.transform = transform
        self.rank = rank
        self.world_size = world_size

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Load variable name ordering
        with open(var_names_path) as f:
            all_var_names = json.load(f)

        var_to_idx = {v: i for i, v in enumerate(all_var_names)}

        self.in_indices = []
        for v in in_vars:
            if v not in var_to_idx:
                raise ValueError(f"Input variable '{v}' not found in var_names.json. "
                               f"Available: {all_var_names}")
            self.in_indices.append(var_to_idx[v])

        self.out_indices = []
        for v in out_vars:
            if v not in var_to_idx:
                raise ValueError(f"Output variable '{v}' not found in var_names.json. "
                               f"Available: {all_var_names}")
            self.out_indices.append(var_to_idx[v])

        # Load or build cached sample index (fast on subsequent runs)
        all_samples = get_or_build_index(
            data_dir, pred_range, subsample, history, rank
        )

        # Shard across ranks
        n_total = len(all_samples)
        samples_per_rank = n_total // world_size

        if samples_per_rank > 0:
            start = rank * samples_per_rank
            end = start + samples_per_rank
            self.samples = all_samples[start:end]
        else:
            self.samples = all_samples

        if rank == 0:
            print(f"[ERA5TimestepDataset] {len(self.samples)} samples/rank "
                  f"(total {n_total}, pred_range={pred_range}, "
                  f"subsample={subsample}, history={history}, "
                  f"in_vars={len(in_vars)}, out_vars={len(out_vars)})", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_fnames, target_fname = self.samples[idx]

        # Load input frames and select channels
        x_frames = []
        for fname in input_fnames:
            path = os.path.join(self.data_dir, fname)
            arr = np.load(path)  # (C_all, H, W) float16
            x_frames.append(arr[self.in_indices])  # (C_in, H, W)
        x = np.concatenate(x_frames, axis=0)  # (C_in * history, H, W)

        # Load target and select channels
        path = os.path.join(self.data_dir, target_fname)
        y = np.load(path)[self.out_indices]  # (C_out, H, W)

        # Convert to tensors
        x = torch.from_numpy(x.copy()).to(self.torch_dtype)
        y = torch.from_numpy(y.copy()).to(self.torch_dtype)

        # Apply normalization transform
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y