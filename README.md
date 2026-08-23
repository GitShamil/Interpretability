# Steering Denoising

Небольшой исследовательский проект для сравнения трёх способов activation
steering в GPT-2 small:

- `baseline`: обычный сдвиг `h + αv`;
- `gaussian`: сдвиг и денойзер, обученный на Gaussian noise;
- `structured`: сдвиг и денойзер, обученный в том числе на train SAE-направлениях.

Для проверки используются 4 отложенных SAE-направления, 30 prompts и 4
значения `alpha`. Concept score выставляется отдельно через LLM. Код считает
только micro `distinct-1/2/3`.

## Установка

Нужен Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[lm,dev]'
```

## Подготовка данных

```bash
python scripts/download_train_data.py
python scripts/download_sae_directions.py
```

После этого:

1. Запишите 4 выбранных SAE ID в `data/validation_direction_ids.json`.
2. Запишите 30 начал текста в `data/prompts.txt`, по одному на строку.

## Запуск

Общие данные для обучения:

```bash
steerlab collect --config configs/collect_gpt2.toml
steerlab split-directions --config configs/collect_gpt2.toml
```

Baseline не требует обучения:

```bash
steerlab evaluate --config configs/evaluate_gpt2.toml --method baseline
```

Gaussian denoiser:

```bash
steerlab train --config configs/train_gaussian.toml
steerlab evaluate --config configs/evaluate_gpt2.toml --method gaussian
```

Structured denoiser:

```bash
steerlab train --config configs/train_structured.toml
steerlab evaluate --config configs/evaluate_gpt2.toml --method structured
```

Каждый evaluation пишет отдельную папку
`artifacts/evaluation/gpt2-layer5/<method>/`:

- `records.jsonl`: `method`, `direction_id`, `alpha`, полный текст;
- `summary.csv`: micro `distinct-1/2/3` для каждой пары direction/alpha.

`records.jsonl` предназначен для внешней LLM-оценки concept score.

## Важные детали

- Интервенция выполняется после шестого блока GPT-2 (`transformer.h.5`).
- SAE-векторы должны лежать в `data/sae_directions.npy` с формой `[N, 768]`.
- Structured denoiser обучается только на направлениях вне validation split.
- `alpha` масштабируется относительно типичного RMS activation, поэтому одна
  сетка применима ко всем направлениям.

## Проверка кода

```bash
pytest
ruff check .
ruff format --check .
```

Описание эксперимента и таблица для результатов находятся в [REPORT.md](REPORT.md).
