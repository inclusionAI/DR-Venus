import importlib.util
from pathlib import Path
import random

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "SFT"
    / "verl"
    / "trainer"
    / "sft_reproducibility.py"
)
SPEC = importlib.util.spec_from_file_location("sft_reproducibility_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reproducibility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproducibility)
derive_data_seed = reproducibility.derive_data_seed
normalize_seed = reproducibility.normalize_seed


def test_derived_data_seeds_are_stable_and_rank_separated():
    assert derive_data_seed(42, rank=3, stream=1) == derive_data_seed(42, rank=3, stream=1)
    assert derive_data_seed(42, rank=3, stream=0) != derive_data_seed(42, rank=3, stream=1)
    assert derive_data_seed(42, rank=2, stream=0) != derive_data_seed(42, rank=3, stream=0)


def test_negative_seed_or_rank_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        normalize_seed(-1)
    with pytest.raises(ValueError, match="rank"):
        derive_data_seed(1, rank=-1)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional in the CPU test environment")
def test_seed_everything_replays_python_numpy_and_torch_rngs():
    import numpy as np
    import torch

    reproducibility.seed_everything(123, deterministic=True)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    reproducibility.seed_everything(123, deterministic=True)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second
