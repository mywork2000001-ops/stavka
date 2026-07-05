# Bukmeker — Multi-Sport Value Betting Engine

[![CI](https://github.com/OWNER/bukmeker/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/bukmeker/actions/workflows/ci.yml)

Математическое и интеграционное ядро, реализованное по спецификации в
[PROMPT.md](PROMPT.md) (дистиллировано из `bukmeker.txt`), упакованное как готовый
к установке Python-пакет. Полная промышленная платформа (Node/React/RabbitMQ/
Postgres/Docker, реальные платежи) в этот пакет не входит — см. «Явные ограничения
текущего объёма» в PROMPT.md.

> Замените `OWNER` в бейдже на имя владельца после публикации на GitHub.

> **Подробная пошаговая инструкция по использованию (установка, примеры кода,
> чтение результатов, подключение реальных данных, частые проблемы) — в
> [USAGE.md](USAGE.md).**

## Возможности

- **Мультиспорт**: футбол (Poisson/Dixon-Coles), баскетбол (Normal margin/total),
  теннис (race-to-N-sets) — единый интерфейс вероятностей исхода для всех трёх.
- **Мультистрана/мультилига**: полный реальный список из 249 стран мира (ISO
  3166-1 через `pycountry`), плюс курируемые реальные лиги/клубы для ~35
  футбольных стран, вся NBA (30 команд) и топ ATP-игроков (см. `entities.py`).
- **Синхронизация через API-ключ**: `bukmeker connector --sync` сливает
  реальные данные, полученные AI-коннектором от любого провайдера, прямо в
  реестр сущностей (лиги/клубы) — идемпотентно, без дублей при повторном запуске.
- **Value detection & bankroll**: EV, Value%, снятие маржи (Shin), фракционный
  Kelly, генератор купонов с ограничением по корреляции исходов.
- **Монетизация купона**: расчёт общего коэффициента, выплаты, комиссии
  платформы и чистой прибыли (только расчёт — без реальных платежей).
- **AI-коннектор данных**: подключите любой источник спортивных данных своим
  API-ключом — Anthropic API реально сопоставляет незнакомую JSON-схему
  провайдера с канонической схемой платформы.
- **Веб-дашборд** (`bukmeker dashboard`): интерактивное приложение в браузере
  поверх той же математики — без командной строки и без чтения кода.

## Установка

```bash
# базовый пакет (математика, мультиспорт, купоны, монетизация)
pip install -e .

# + AI-коннектор данных (Anthropic SDK + requests)
pip install -e ".[connectors]"

# + веб-дашборд (Streamlit)
pip install -e ".[dashboard]"

# + инструменты разработки (pytest, ruff) — включает connectors и dashboard
pip install -e ".[dev]"
```

## Структура

```
bukmeker/
  pyproject.toml    — сборка (setuptools), entry point `bukmeker`, extras [connectors, dashboard, dev]
  USAGE.md           — пошаговая инструкция по использованию
  .github/workflows/ci.yml — CI: ruff + pytest на Python 3.11 и 3.12
  bukmeker/
    features.py       — экспоненциальное затухание, EMA, взвешенное скользящее среднее
    ratings.py         — Elo, байесовский рейтинг, Poisson attack/defence (MLE)
    entities.py        — Sport/Country/League/Competitor + seed-реестр (249 реальных
                          стран через pycountry, ~35 футбольных лиг, вся NBA, топ ATP)
    models/goals.py    — Poisson, бивариатный Poisson, Dixon-Coles, Negative Binomial,
                          Skellam, Monte-Carlo
    sports/
      football.py        — predict_1x2 (обёртка над Dixon-Coles)
      basketball.py       — predict_moneyline / predict_spread_cover / predict_total
      tennis.py            — race_to_win_prob, predict_match_win_prob (best_of 3/5)
    margin.py          — снятие маржи букмекера (Shin, мультипликативный)
    calibration.py     — Platt scaling, isotonic regression, ECE/Brier/LogLoss/ROC-AUC
    value_betting.py   — EV, Value%, Overlay, Probability Edge, bootstrap CI, Kelly
    coupon.py          — генератор купонов с ограничением по корреляции исходов
    monetization.py    — CouponReport: общий коэффициент, комиссия, чистая выплата
    connectors/
      schema.py           — CanonicalMatch, FieldMapping, get_by_path, find_record_list
      raw_source.py        — RawDataSource: generic HTTP-клиент по API-ключу
      ai_mapper.py          — ClaudeFieldMapper: реальный вызов Anthropic API
      ai_connector.py        — AIDataConnector: fetch + normalize
      sync.py                — sync_registry_from_matches: слияние live-данных в реестр
    dashboard.py        — Streamlit-дашборд (все вкладки: спорт/купон/сущности)
    cli.py             — `bukmeker demo`, `bukmeker connector [--sync]`, `bukmeker dashboard`
  tests/               — 130 unit-тестов; сетевые/AI-вызовы и дашборд протестированы
                          через инжектируемые фейки и headless AppTest, без реальных
                          запросов, трат или браузера
  demo.py              — тонкая обёртка над bukmeker.cli.main(["demo"])
  dashboard_app.py     — точка входа для `streamlit run dashboard_app.py`
```

## Запуск

```bash
pytest tests/ -q        # 130 passed
bukmeker demo             # сквозная синтетическая демонстрация (после pip install -e .)
bukmeker dashboard         # веб-дашборд в браузере (http://localhost:8501)
python demo.py             # то же самое без установки пакета
ruff check .                # линт (используется в CI)
```

## Сквозной пример (`bukmeker demo`)

Прогоняет полный конвейер на synthetic-данных: подгонка Poisson attack/defence
рейтингов → ожидаемые голы конкретного матча → матрица Dixon-Coles → 1X2
вероятности → снятие маржи букмекера (Shin) → обнаружение value bet (EV, Value%) →
half-Kelly стейк → генерация купона из нескольких матчей с фильтрацией по
корреляции → баскетбол и теннис тем же движком → монетизация лучшего купона.

## Веб-дашборд (`bukmeker dashboard`)

```bash
pip install -e ".[dashboard]"
bukmeker dashboard              # открывает http://localhost:8501 в браузере
```

Слева — меню навигации (не плоские вкладки) с иконками и понятными названиями:
**❓ Справка** (глоссарий терминов и как пользоваться — открывается первой),
**⚽ Футбол / 🏀 Баскетбол / 🎾 Теннис** (вероятности исхода на ползунках и
полях ввода коэффициентов, с расчётом EV/Value%/Kelly — у каждого поля
всплывающая подсказка), **🎫 Купон и монетизация** (редактируемая таблица
ставок → генерация купонов → отчёт по выплате), **🌍 Страны и лиги** (все 249
реальных стран мира + честная метка, для каких из них есть данные по лигам/
клубам, а для каких — нет) и **ℹ️ О проекте** (границы объёма). Это тот же
движок, что и в `bukmeker demo` — просто с виджетами и подсказками вместо чтения кода.

По умолчанию дашборд открыт без пароля (с предупреждением на экране) — это
режим для локального использования на своей машине. Чтобы закрыть доступ
паролем (например, перед тем как открыть порт наружу):

```bash
export BUKMEKER_DASHBOARD_PASSWORD="ваш_пароль"
bukmeker dashboard
```

Это общий пароль на всё приложение (не отдельные аккаунты пользователей) —
простейшая защита, а не полноценная многопользовательская авторизация.

## AI-коннектор данных (`bukmeker connector`)

Реальная (не синтетическая) интеграция: требует API-ключ источника данных и
Anthropic API-ключ. Тратит реальные деньги и делает реальные сетевые запросы.

```bash
export SOURCE_API_KEY="..."       # ключ вашего провайдера спортивных данных
export ANTHROPIC_API_KEY="..."    # ваш ключ Anthropic API

bukmeker connector \
  --source-url https://api.example-sports-data.com \
  --path fixtures?date=2026-07-05 \
  --key-location header --key-name x-api-key
```

Без ключей команда печатает usage и завершается кодом 1 — синтетического
fallback намеренно нет, чтобы не создавать иллюзию, что показаны реальные данные.

Программно:

```python
from bukmeker.connectors import AIDataConnector, ClaudeFieldMapper, RawDataSource

source = RawDataSource(base_url="https://api.example.com", api_key="SOURCE_KEY")
mapper = ClaudeFieldMapper(api_key="ANTHROPIC_KEY")
matches = AIDataConnector(source, mapper).fetch_and_normalize("fixtures")
```

### Синхронизация реестра сущностей (`--sync`)

`bukmeker connector --sync` дополнительно сливает полученные лиги/команды в
реестр сущностей (`bukmeker.entities`) — это и есть механизм пополнения
системы реальными данными через API-ключ, вместо ручного добавления записей
в код:

```bash
bukmeker connector --source-url ... --path fixtures \
  --sync --sync-sport Football --sync-country GBR
```

Идемпотентно (повторный запуск с теми же данными не создаёт дублей: лиги
матчатся по названию+виду спорта, команды — по названию+лиге). `--sync-country`
нужен потому, что данные о матчах обычно не содержат страну лиги — без этого
флага новую лигу было бы не к чему честно привязать. Программно:

```python
from bukmeker.connectors import sync_registry_from_matches
from bukmeker.entities import build_seed_registry

registry = build_seed_registry()
report = sync_registry_from_matches(registry, matches, sport_id=1, fallback_country_id=usa_country_id)
print(report.leagues_added, report.competitors_added)
```

## Монетизация купона

```python
from bukmeker.coupon import generate_coupons
from bukmeker.monetization import build_coupon_report, format_coupon_report

coupons = generate_coupons(value_bets, bankroll=10_000)
report = build_coupon_report(coupons[0], stake=100.0, platform_fee_pct=0.05)
print(format_coupon_report(report))
```

## Область проекта

См. «Явные ограничения текущего объёма» в [PROMPT.md](PROMPT.md): без backend/БД/
очередей/фронтенда/инфраструктуры, без реальных платежей. Список стран — реальный
и полный (249, ISO 3166-1); лиги/клубы — курируемая, реальная, но не исчерпывающая
подборка (полное покрытие по конкретному провайдеру — через `--sync`).
