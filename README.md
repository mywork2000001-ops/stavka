# Bukmeker — Value Betting Math Core

[![CI](https://github.com/OWNER/bukmeker/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/bukmeker/actions/workflows/ci.yml)

Математическое ядро, реализованное по спецификации в [PROMPT.md](PROMPT.md)
(дистиллировано из `bukmeker.txt`), упакованное как готовый к установке Python-пакет.
Полная промышленная платформа (Node/React/RabbitMQ/Postgres/Docker) в этот пакет
не входит — здесь только проверяемая тестами количественная логика.

> Замените `OWNER` в бейдже на имя владельца после публикации на GitHub.

## Установка

```bash
# из корня проекта
pip install -e ".[dev]"
```

## Структура

```
bukmeker/
  pyproject.toml    — сборка (setuptools), entry point `bukmeker`, extras [dev]
  .github/workflows/ci.yml — CI: ruff + pytest на Python 3.11 и 3.12
  bukmeker/
    features.py     — экспоненциальное затухание, EMA, взвешенное скользящее среднее
    ratings.py       — Elo, байесовский рейтинг, Poisson attack/defence (MLE)
    models/goals.py  — Poisson, бивариатный Poisson, Dixon-Coles, Negative Binomial,
                        Skellam, Monte-Carlo
    margin.py        — снятие маржи букмекера (Shin, мультипликативный)
    calibration.py   — Platt scaling, isotonic regression, ECE/Brier/LogLoss/ROC-AUC
    value_betting.py — EV, Value%, Overlay, Probability Edge, bootstrap CI, Kelly
    coupon.py        — генератор купонов с ограничением по корреляции исходов
    cli.py           — точка входа `bukmeker demo`
  tests/             — 55 unit-тестов, каждая формула проверена против scipy/sklearn
                        или независимого расчёта
  demo.py            — тонкая обёртка над bukmeker.cli.run_demo() для запуска без установки
```

## Запуск

```bash
pytest tests/ -q        # 55 passed
bukmeker demo            # сквозная демонстрация (после pip install -e .)
python demo.py            # то же самое без установки пакета
ruff check .               # линт (используется в CI)
```

## Сквозной пример

`bukmeker demo` прогоняет полный конвейер на synthetic-данных: подгонка Poisson
attack/defence рейтингов → ожидаемые голы конкретного матча → матрица Dixon-Coles →
1X2 вероятности → снятие маржи букмекера (Shin) → обнаружение value bet (EV, Value%) →
half-Kelly стейк → генерация купона из нескольких матчей с фильтрацией по корреляции.

## Область проекта

Реализовано осознанно только математическое ядро — см. раздел «Явные ограничения
текущего объёма» в [PROMPT.md](PROMPT.md) для полного списка того, что намеренно
не входит (backend, БД, очереди, фронтенд, инфраструктура).
