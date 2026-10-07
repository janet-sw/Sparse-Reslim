from pathlib import Path
import sys

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from config_utils import batch_size_per_rank, load_yaml_config


def test_table1_config_expands_environment(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/tmp/era5")
    monkeypatch.setenv("OUTPUT_ROOT", "/tmp/results")
    monkeypatch.setenv("TRAINING_SEED", "43")

    config = load_yaml_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "table1_sparse_era5_1.40625.yaml"
    )

    assert config["data"]["era5_dir"] == "/tmp/era5"
    assert config["trainer"]["seed"] == "43"
    assert config["trainer"]["checkpoint_save_path"].endswith("seed_43")
    assert batch_size_per_rank(config, 16) == 2


def test_table1_config_uses_documented_defaults(monkeypatch):
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("TRAINING_SEED", raising=False)

    config = load_yaml_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "table1_dense_era5_1.40625.yaml"
    )

    assert config["trainer"]["seed"] == "42"
    assert config["trainer"]["checkpoint_save_path"].startswith("./outputs/")


def test_global_batch_size_must_divide_world_size():
    with pytest.raises(ValueError, match="not divisible"):
        batch_size_per_rank({"trainer": {"batch_size": 31}}, 16)
