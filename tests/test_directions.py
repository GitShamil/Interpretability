from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from steering_denoising.directions import DirectionBank, DirectionSplitManifest
from steering_denoising.exceptions import CompatibilityError


def test_direction_bank_load_and_subset(tmp_path: Path) -> None:
    path = tmp_path / "directions.npy"
    np.save(path, np.asarray([[3.0, 4.0], [0.0, -2.0], [1.0, 1.0]]))
    bank = DirectionBank.load(path, expected_dim=2)

    torch.testing.assert_close(bank.vectors.norm(dim=1), torch.ones(3))
    subset = bank.subset_ids([2, 0])
    assert subset.ids == (2, 0)
    assert subset.d_model == 2


def test_explicit_validation_split_is_compact_and_bound_to_bank(tmp_path: Path) -> None:
    bank = DirectionBank(torch.eye(6))
    split = DirectionSplitManifest.create(bank, validation_ids=[4, 5])
    path = split.save(tmp_path / "split.json")
    loaded = DirectionSplitManifest.load(path, bank=bank)

    assert loaded == split
    assert loaded.train_ids == (0, 1, 2, 3)
    assert json.loads(path.read_text()) == {
        "bank_hash": bank.bank_hash,
        "validation_ids": [4, 5],
    }

    changed = DirectionBank(torch.eye(6) + 0.01)
    with pytest.raises(CompatibilityError, match="another SAE bank"):
        DirectionSplitManifest.load(path, bank=changed)
