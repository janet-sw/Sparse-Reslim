# Sparse-Reslim

Official implementation of **Less Tokens, Better Forecasts: Sparse Residual
Routing for Efficient Weather Prediction** (ECCV 2026).

[Paper](https://arxiv.org/abs/2607.02829) ·
[Primary repository](https://github.com/janet-sw/Sparse-Reslim)

Sparse-Reslim is a parameter-free routing module for dense weather prediction.
It sends only a subset of spatial tokens through the expensive middle
Transformer blocks, computes their residual updates, and scatters only those
updates back to the full sequence. Every grid position is preserved.

## Highlights

- Dense–sparse–dense block schedules with random, parameter-free token routing.
- Residual-delta reconstruction: unrouted tokens remain exactly unchanged.
- Deterministic Res-Slim-ViT and generative EDM implementations.
- Asymmetric sparse EDM attention: sparse noisy-state queries attend to the
  complete conditioning sequence.
- Standard PyTorch attention works without distributed initialization;
  xFormers, MPI, and DDStore are optional HPC features.

## Repository layout

```text
configs/                  experiment configurations
examples/                 deterministic and generative training entry points
src/climate_learn/        models, data loaders, metrics, and utilities
tests/                    unit and model smoke tests
```

The main model implementations are:

- `src/climate_learn/models/hub/res_slimvit_adaptive.py`
- `src/climate_learn/models/hub/edm.py`

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build suitable for your
accelerator first, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

xFormers is only required when `FusedAttn.CK` is selected. MPI and DDStore are
only required when `ORBIT_USE_DDSTORE=1`; standard data loading does not
require them. PyTorch Lightning is not used by the forecasting entry points;
install `.[climatebench]` only when using the legacy ClimateBench module.

## Quick verification

```bash
pytest tests/models/test_sparse_reslim.py -q
```

The smoke tests instantiate compact deterministic and EDM models, verify the
sparse-stage token counts, and run forward and backward passes on CPU.

## Training

Prepare ERA5 in the ClimateLearn NPZ layout, then replace `<DATA_DIR>` and
`<OUTPUT_DIR>` in the selected configuration.

Deterministic model:

```bash
srun python examples/era5_forecasting_sparsity.py \
  configs/sparse_reslim_deterministic_era5.yaml
```

Generative EDM model:

```bash
srun python examples/train_sparse_reslim_edm.py \
  configs/sparse_reslim_edm_era5.yaml
```

The paper configurations use `keep_ratio: 0.25`. Set `keep_ratio: 1.0` for the
dense baseline. The deterministic model uses a `(2, 8, 2)` block split; the
180M EDM uses `(2, 4, 2)`.

The training entry points use the Slurm environment plus PyTorch
distributed/FSDP, matching the paper runs. Model import and single-process
forward execution do not require distributed initialization.

### Table 1 deterministic setup

The paired configs below encode the paper's 1.40625° dense and Sparse-Reslim
settings: a 128 × 256 grid, 120-hour lead time, global batch size 32, 30 epochs,
bfloat16, activation checkpointing, and FSDP over 16 MI250X GCDs.

```bash
export DATA_ROOT=/path/to/era5_1.40625
export OUTPUT_ROOT=$PWD/outputs
export TRAINING_SEED=42

# Sparse-Reslim (r=0.25, block split 2/8/2)
sbatch examples/launch_table1_frontier.sh

# Dense baseline, using the same seed and training protocol
CONFIG_PATH=$PWD/configs/table1_dense_era5_1.40625.yaml \
  sbatch examples/launch_table1_frontier.sh
```

`history`, `window`, `subsample`, paths, seed, and activation checkpointing are
all controlled by YAML. The batch size in YAML is global (2 samples per rank
for the 16-GPU Table 1 runs). For a controlled comparison, run dense and sparse
with identical seeds; multiple seeds are recommended because random routing
introduces run-to-run variation.

## Data and checkpoints

ERA5 data and trained checkpoints are not redistributed in this repository.
Configuration files document the expected variables, temporal split, forecast
lead time, and model hyperparameters. The Table 1 configs reproduce the stated
training protocol but do not guarantee bitwise-identical metrics across
software stacks or random seeds.

## Acknowledgements

This code builds on
[ORBIT-2](https://github.com/XiaoWang-Github/ORBIT-2) and
[ClimateLearn](https://github.com/aditya-grover/climate-learn).

## Citation

```bibtex
@inproceedings{wang2026sparsereslim,
  title     = {Less Tokens, Better Forecasts: Sparse Residual Routing for Efficient Weather Prediction},
  author    = {Wang, Janet and Zhang, Yunbei and Zhao, Lin and Xiao, Xi and Hamm, Jihun and Wang, Xiao},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
