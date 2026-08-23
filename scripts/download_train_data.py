#!/usr/bin/env python3
"""Stream one million GPT-2 tokens from OpenWebText into data/train.txt."""

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_ID = "Skylion007/openwebtext"
DATASET_REVISION = "c2297e5d464b5905ee2440e6f5fc175095e55307"
TOKENIZER_ID = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TARGET_TOKENS = 1_000_000
CHUNK_TOKENS = 256
OUTPUT = Path("data/train.txt")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} already exists")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    dataset = load_dataset(
        DATASET_ID,
        name="plain_text",
        split="train",
        revision=DATASET_REVISION,
        streaming=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".txt.part")
    usable_tokens = 0
    lines = 0
    rows = iter(dataset)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            token_ids = tokenizer.encode(row["text"], add_special_tokens=False, verbose=False)
            for start in range(0, len(token_ids), CHUNK_TOKENS):
                line = " ".join(tokenizer.decode(token_ids[start : start + CHUNK_TOKENS]).split())
                if not line:
                    continue
                handle.write(line + "\n")
                usable_tokens += min(
                    len(tokenizer.encode(line, add_special_tokens=False)), CHUNK_TOKENS
                )
                lines += 1
                if usable_tokens >= TARGET_TOKENS:
                    break
            if usable_tokens >= TARGET_TOKENS:
                break
    rows.close()

    temporary.replace(OUTPUT)
    print(f"Saved {usable_tokens:,} usable tokens in {lines:,} lines to {OUTPUT}")


if __name__ == "__main__":
    main()
