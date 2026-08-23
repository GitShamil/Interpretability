# Данные

Из корня репозитория запустите:

```bash
python scripts/download_train_data.py
python scripts/download_sae_directions.py
```

В этой папке нужны четыре файла:

- `train.txt` — около 1 млн GPT-2 токенов, одна запись на строку;
- `sae_directions.npy` — SAE decoder vectors формы `[N, 768]`;
- `validation_direction_ids.json` — JSON-массив из 4 выбранных feature ID;
- `prompts.txt` — 30 начал текста, по одному на строку.

Запишите свои ID в `validation_direction_ids.json` и не меняйте их после
обучения structured denoiser.
