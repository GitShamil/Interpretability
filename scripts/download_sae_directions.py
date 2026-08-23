#!/usr/bin/env python3
"""Download layer-5 GPT-2 SAE decoder directions."""

from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs"
REVISION = "21df74db7b9f690cd2403a20485cf75f46330d0d"
WEIGHTS_FILE = "v5_32k_layer_5.pt/sae_weights.safetensors"
OUTPUT = Path("data/sae_directions.npy")
EXPECTED_SHAPE = (32768, 768)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} already exists")

    weights_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=WEIGHTS_FILE,
        revision=REVISION,
    )
    directions = load_file(weights_path, device="cpu")["W_dec"].float().numpy()
    if directions.shape != EXPECTED_SHAPE:
        raise ValueError(f"expected {EXPECTED_SHAPE}, got {directions.shape}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".npy.part")
    with temporary.open("wb") as handle:
        np.save(handle, directions, allow_pickle=False)
    temporary.replace(OUTPUT)
    print(f"Saved {directions.shape} directions to {OUTPUT}")


if __name__ == "__main__":
    main()
