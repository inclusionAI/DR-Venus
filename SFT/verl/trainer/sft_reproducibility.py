"""Deterministic seed plumbing for the SFT trainer.

The model seed is shared by all ranks so replicated/FSDP model construction is
identical.  Data-loader streams are derived from the data-parallel rank and
train/validation stream so workers do not accidentally share RNG state.
"""

from __future__ import annotations

import functools
import os
import random


_MAX_TORCH_SEED = 2**63 - 1


def normalize_seed(seed: int) -> int:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed % _MAX_TORCH_SEED


def derive_data_seed(seed: int, *, rank: int, stream: int = 0) -> int:
    if rank < 0 or stream < 0:
        raise ValueError("rank and stream must be non-negative")
    # Large odd constants keep neighbouring ranks/streams separated while the
    # modulo makes the result acceptable to torch.Generator.manual_seed.
    return (normalize_seed(seed) + rank * 1_000_003 + stream * 10_000_019) % _MAX_TORCH_SEED


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, torch and every visible CUDA device."""

    import numpy as np
    import torch

    seed = normalize_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    return seed


def seed_data_worker(worker_id: int, *, base_seed: int) -> None:
    import numpy as np
    import torch

    worker_seed = (normalize_seed(base_seed) + int(worker_id)) % _MAX_TORCH_SEED
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def dataloader_seed_components(seed: int, *, rank: int, stream: int = 0):
    """Return a seeded torch Generator and a picklable worker initializer."""

    import torch

    derived = derive_data_seed(seed, rank=rank, stream=stream)
    generator = torch.Generator()
    generator.manual_seed(derived)
    worker_init_fn = functools.partial(seed_data_worker, base_seed=derived)
    return generator, worker_init_fn
