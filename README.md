# Bukmeker — Multi-Sport Value Betting Engine

[![CI](https://github.com/mywork2000001-ops/stavka/actions/workflows/ci.yml/badge.svg)](https://github.com/mywork2000001-ops/stavka/actions/workflows/ci.yml)

Математическое и интеграционное ядро, реализованное по спецификации в
[PROMPT.md](PROMPT.md) (дистиллировано из `bukmeker.txt`), упакованное как готовый
к установке Python-пакет. Полная промышленная платформа (Node/React/RabbitMQ/
Postgres/Docker, реальные платежи) в этот пакет не входит — см. «Явные ограничения
текущего объёма» в PROMPT.md.

Репозиторий: [github.com/mywork2000001-ops/stavka](https://github.com/mywork2000001-ops/stavka)

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
- **Реальный прогноз, а не только калькулятор**: обучение `PoissonStrength` на
  реальной истории результатов, обязательный бэктест на отложенных данных
  (ROC-AUC/ECE/Log Loss/Brier) до какого-либо использования, и автоматическое
  сканирование предстоящих коэффициентов на value bets → готовый купон, без
  единого вручную введённого числа (`bukmeker.backtest`, `bukmeker.live_predictions`).
- **Веб-дашборд** (`bukmeker dashboard`): интерактивное приложение в браузере
  поверх той же математики — без командной строки и без чтения кода.
- **Paper trading**: журнал прогнозов против реальных исходов (без реальных
  ставок) с CLV (Closing Line Value), win rate и ROI — способ проверить модель
  на новых матчах со временем, прежде чем рисковать деньгами (`bukmeker.paper_trading`,
  вкладка «📝 Paper Trading» в дашборде). Хранится в JSON-файле, а не в сессии
  браузера — переживает перезапуски.

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
  tests/               — 233 unit-тестов; сетевые/AI-вызовы и дашборд протестированы
                          через инжектируемые фейки и headless AppTest, без реальных
                          запросов, трат или браузера
  demo.py              — тонкая обёртка над bukmeker.cli.main(["demo"])
  dashboard_app.py     — точка входа для `streamlit run dashboard_app.py`
```

## Запуск

```bash
pytest tests/ -q        # 233 passed
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
**⚽ Футбол / 🏀 Баскетбол / 🎾 Теннис** (реальные названия команд/игроков из
реестра сущностей — например, "Arsenal — Chelsea" — вероятности исхода на
ползунках и полях ввода коэффициентов, с расчётом EV/Value%/Kelly), **🎫 Купон
и монетизация** (гид "выбор вида спорта → чемпионат → команды → анализ" тем же
движком, что и отдельные страницы спорта, плюс ручной ввод для опытных →
генерация купонов → отчёт по выплате → сохранение в историю и статистика по
периодам "Сегодня/Этот месяц/Всё время": число купонов, сумма ставок, чистая
прибыль, ROI), **🌍 Страны и лиги** (все 249 реальных
стран мира + честная метка, для каких из них есть данные по лигам/клубам, а
для каких — нет), **🔌 AI-коннектор** (ввод API-ключа источника данных и
Anthropic-ключа/версии модели прямо в браузере — единственная страница с
реальными сетевыми и платными запросами) и **ℹ️ О проекте** (границы объёма).
Это тот же движок, что и в `bukmeker demo` — просто с виджетами и подсказками
вместо чтения кода.

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

## Обучение модели, бэктест и автогенерация купонов

Закрывает разрыв между "калькулятором EV/Kelly от вручную введённой вероятности"
и настоящей прогнозной системой. Три шага, каждый — реальный код, не демонстрация:

```python
from bukmeker.connectors import AIDataConnector, ClaudeFieldMapper, RawDataSource
from bukmeker.connectors.historical import HISTORICAL_FIELDS, apply_historical_mapping, to_rating_arrays
from bukmeker.backtest import backtest_poisson_ratings
from bukmeker.live_predictions import scan_fixtures_for_value
from bukmeker.coupon import generate_coupons

# 1) Реальная история результатов -> реальный рейтинг
source = RawDataSource(base_url="https://api.provider.com", api_key="SOURCE_KEY")
mapper = ClaudeFieldMapper(api_key="ANTHROPIC_KEY", target_fields=HISTORICAL_FIELDS)
connector = AIDataConnector(source, mapper, apply_fn=apply_historical_mapping)
historical_matches = connector.fetch_and_normalize("results")

home_ids, away_ids, home_goals, away_goals, teams = to_rating_arrays(historical_matches)

# 2) ОБЯЗАТЕЛЬНЫЙ бэктест ДО использования модели
fitted, metrics = backtest_poisson_ratings(home_ids, away_ids, home_goals, away_goals, teams)
print(metrics)  # log_loss, brier, roc_auc, ece, n_train, n_test
assert metrics["roc_auc"] > 0.55, "модель не показывает предсказательной силы"

# 3) Реальные предстоящие матчи + коэффициенты -> автоматически найденные value bets -> купон
fixtures_mapper = ClaudeFieldMapper(api_key="ANTHROPIC_KEY")  # схема коэффициентов (по умолчанию)
fixtures = AIDataConnector(source, fixtures_mapper).fetch_and_normalize("fixtures")

candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1, min_ev=0.0)
coupons = generate_coupons(candidates, bankroll=10_000)
```

То же самое — в дашборде: страница **🔌 AI-коннектор** → разделы «📈 Обучение
модели на истории результатов» (с явным предупреждением при слабом ROC-AUC) и
«🤖 Автоматическая генерация купонов».

**Честная граница**: успешный бэктест на истории — не гарантия будущей
прибыли (рынки меняются, короткие серии определяются дисперсией сильнее,
чем качеством модели) и реализовано пока только для футбола (`PoissonStrength`).

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

## Аудит и качество

[AUDIT_PROMPT.md](AUDIT_PROMPT.md) — независимый аудит проекта по корректности,
безопасности, покрытию тестами и согласованности документации, с журналом всех
найденных и исправленных дефектов (найдено и закрыто 12 реальных проблем на
момент последнего прохода).
