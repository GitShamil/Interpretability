"""Small reproducibility and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from steering_denoising.exceptions import ConfigurationError


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 63-bit seed independent of Python's randomized hash()."""

    digest = hashlib.sha256(str(base_seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & (2**63 - 1)


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ConfigurationError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        dtype = mapping[value]
    except KeyError as error:
        raise ConfigurationError(f"Unsupported dtype: {value}") from error
    if device.type == "cpu" and dtype == torch.float16:
        raise ConfigurationError("float16 on CPU is unsupported for this pipeline; use float32.")
    return dtype


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()
