# Инструкция по использованию

Пошаговое руководство для тех, кто хочет реально запустить и использовать
`bukmeker` — от установки до подключения живых данных. Техническое описание
архитектуры и принятых решений — в [PROMPT.md](PROMPT.md); краткий обзор
возможностей — в [README.md](README.md). Здесь — практика: что вводить и как
читать результат.

## Содержание

1. [Что нужно перед началом](#1-что-нужно-перед-началом)
2. [Установка](#2-установка)
3. [Быстрый старт: `bukmeker demo`](#3-быстрый-старт-bukmeker-demo)
4. [Как читать результаты](#4-как-читать-результаты)
5. [Веб-дашборд: `bukmeker dashboard`](#5-веб-дашборд-bukmeker-dashboard)
6. [Использование по частям (Python API)](#6-использование-по-частям-python-api)
7. [Мультиспорт и сущности (страны/лиги/клубы)](#7-мультиспорт-и-сущности)
8. [Монетизация купона](#8-монетизация-купона)
9. [Подключение реальных данных через AI-коннектор](#9-подключение-реальных-данных-через-ai-коннектор)
10. [Тестирование и разработка](#10-тестирование-и-разработка)
11. [Частые проблемы](#11-частые-проблемы)

---

## 1. Что нужно перед началом

- Python 3.11 или новее (`python --version`).
- pip (идёт вместе с Python).
- Для раздела 9 (AI-коннектор) — ключ любого провайдера спортивных данных
  (например, API-Football, TheOddsAPI и т.п.) и ключ Anthropic API
  (console.anthropic.com). Без них работает всё, кроме `bukmeker connector`.

## 2. Установка

Из корня проекта (`bukmeker/`, там где лежит `pyproject.toml`):

```bash
# базовый пакет — вся математика, мультиспорт, купоны, монетизация
pip install -e .

# + AI-коннектор данных (добавляет anthropic и requests)
pip install -e ".[connectors]"

# + веб-дашборд (Streamlit)
pip install -e ".[dashboard]"

# + инструменты разработки (pytest, ruff) — включает connectors и dashboard
pip install -e ".[dev]"
```

Проверка, что всё установилось:

```bash
bukmeker --help
```

## 3. Быстрый старт: `bukmeker demo`

```bash
bukmeker demo
```

(или `python demo.py`, если пакет не устанавливали). Команда ничего не
скачивает из интернета и не тратит деньги — все данные внутри синтетические,
это витрина того, что умеет движок. Она печатает 8 пронумерованных разделов
подряд: рейтинги команд → ожидаемые голы → вероятности исхода → снятие маржи
→ value bet → купон → баскетбол/теннис → монетизация купона.

## 4. Как читать результаты

Пример строки из раздела 5 (`VALUE DETECTION`):

```
home_win  p_model=0.6838 odds=1.75 EV=+0.1967 Value%=+19.67% HalfKellyStake=1311.47  <-- VALUE BET
```

- **p_model** — вероятность исхода по модели (0.68 = 68%).
- **odds** — коэффициент букмекера на этот исход.
- **EV** (Expected Value) — ожидаемая прибыль на 1 единицу ставки. `+0.1967`
  значит: в среднем на длинной дистанции такая ставка приносит +19.67% от
  размера ставки. `EV > 0` — потенциально выгодная ставка, `EV <= 0` —
  невыгодная, движок её отбросит.
- **Value%** — то же самое ожидаемое преимущество, выраженное как
  `p_model * odds - 1`, тот же знак и смысл, что EV.
- **HalfKellyStake** — сколько поставить (в единицах банкролла), по
  половинному критерию Келли — консервативнее полного Kelly, меньше риск
  просадки банка.
- Пометка `<-- VALUE BET` — модель считает эту ставку недооценённой рынком.

Раздел 4 (снятие маржи):

```
Shin fair probabilities: home=0.5471 draw=0.2584 away=0.1946
```

Это "справедливые" вероятности — то, что осталось после вычитания прибыли
букмекера (маржи) из его коэффициентов методом Шина. EV в разделе 5 считается
именно от `p_model`, а не от `fair_probs` напрямую — модель и рынок
сравниваются по коэффициенту, а fair_probs используются для диагностики,
насколько сильно рынок расходится с моделью (`overlay`, `probability_edge`,
см. раздел 6.5 ниже).

Раздел 8 (монетизация в bukmeker demo):

```
gross_payout: 570.00
platform_fee (5%): 28.50
net_payout: 541.50
net_profit: 441.50
```

При ставке 100 и общем коэффициенте купона 5.70: вы получите 570 при
выигрыше, платформа берёт комиссию 5% от выплаты (28.50), вам остаётся
541.50, чистая прибыль сверх ставки — 441.50. **Это расчёт, не реальный
платёж** — денег никто никуда не переводит.

## 5. Веб-дашборд: `bukmeker dashboard`

Если не хочется читать вывод в терминале или писать код — есть интерактивное
веб-приложение поверх того же движка.

```bash
pip install -e ".[dashboard]"
bukmeker dashboard
```

Откроется браузер на `http://localhost:8501`. Другой порт: `bukmeker dashboard
--port 8600`. Слева — меню навигации (не плоские вкладки) с иконками:

- **❓ Справка** — открывается первой: что вообще делает приложение, что
  означает каждый пункт меню, и глоссарий терминов (EV, Kelly, overround,
  λ, ρ и т.д.) в таблице. С этого стоит начать, если непонятно, куда жать.
- **⚽ Футбол** — ползунки ожидаемых голов и ρ (Dixon-Coles), поля коэффициентов
  1X2 → график вероятностей, справедливые вероятности (Shin), таблица
  EV/Value%/Kelly для каждого исхода.
- **🏀 Баскетбол** — ожидаемая разница очков/std, коэффициенты на победу,
  moneyline-вероятности + линия тотала (over/under).
- **🎾 Теннис** — вероятность выигрыша сета, формат матча (best of 3/5),
  коэффициенты на игроков.
- **🎫 Купон и монетизация** — редактируемая таблица ставок (можно добавлять/
  удалять строки, есть подсказка «Как заполнять таблицу?»), кнопка
  «Сгенерировать купоны» → таблица комбинаций с положительным EV → выбор
  купона, ставка, комиссия платформы → выплата.
- **🌍 Страны и лиги** — сверху честные метрики (249 реальных стран vs.
  сколько из них имеют данные по лигам); два режима просмотра: «по виду
  спорта» (как раньше) и «по стране» — поиск по всем 249 странам мира с явным
  сообщением, если данных для выбранной страны ещё нет.
- **ℹ️ О проекте** — границы объёма (что не реализовано и почему).

У каждого технического поля (ползунка, числового ввода) есть значок `(?)` —
наведите курсор, чтобы увидеть подсказку простым языком, не заглядывая в этот
документ.

Приложение не делает сетевых запросов и не тратит деньги — данные внутри
синтетические/из seed-реестра, как и в `bukmeker demo`. Для остановки —
`Ctrl+C` в терминале, где запущена команда.

Если команда `streamlit` не найдена — установите extras: `pip install -e
".[dashboard]"`.

### Пароль на дашборд

По умолчанию `bukmeker dashboard` открыт без пароля — на экране виден жёлтый
предупреждающий баннер об этом. Чтобы закрыть доступ (например, если порт
будет виден не только вам), задайте пароль до запуска:

```bash
export BUKMEKER_DASHBOARD_PASSWORD="ваш_пароль"
bukmeker dashboard
```

Теперь при открытии приложения сначала появится экран входа; неверный пароль
показывает ошибку и не пускает дальше, верный — открывает все вкладки на всё
время сессии браузера. Это один общий пароль на всё приложение (не отдельные
аккаунты пользователей с разделением банкролла/купонов).

## 6. Использование по частям (Python API)

### 6.1 Рейтинги команд

```python
from bukmeker.ratings import PoissonStrength
import numpy as np

# home_ids/away_ids/home_goals/away_goals — ваша историческая база матчей
strength = PoissonStrength.fit(home_ids, away_ids, home_goals, away_goals, teams=list_of_team_names)
lam_home, lam_away = strength.expected_goals("Arsenal", "Everton")  # ожидаемые голы
```

Альтернативы: `elo_expected_score`/`elo_update` (простой Elo) и
`BayesianRating` (гауссово обновление убеждения о силе команды) в том же
модуле — используйте, если исторических данных мало для MLE-подгонки Poisson.

### 6.2 Вероятности исхода (футбол)

```python
from bukmeker.sports.football import predict_1x2

probs = predict_1x2(home_lambda=lam_home, away_lambda=lam_away, rho=-0.08)
# {'home_win': 0.68, 'draw': 0.18, 'away_win': 0.13}
```

### 6.3 Снятие маржи букмекера

```python
from bukmeker.margin import shin_margin_removal
import numpy as np

fair_probs = shin_margin_removal(np.array([1.75, 3.60, 4.75]))  # [home, draw, away]
```

Если коэффициенты не содержат маржи (сумма implied-вероятностей ≤ 1),
функция бросит `ValueError` — это ожидаемое поведение, не баг.

### 6.4 Калибровка вероятностей модели

```python
from bukmeker.calibration import IsotonicScaler, calculate_metrics

scaler = IsotonicScaler().fit(raw_model_scores, y_true)
calibrated_probs = scaler.predict_proba(raw_model_scores)
metrics = calculate_metrics(y_true, calibrated_probs)  # log_loss, brier, roc_auc, ece
```

### 6.5 Value detection и банкролл

```python
from bukmeker.value_betting import expected_value, value_percentage, kelly_stake

ev = expected_value(model_prob=0.68, odds=1.75)
stake = kelly_stake(bankroll=10_000, prob=0.68, odds=1.75, fraction=0.5)  # half-Kelly
```

### 6.6 Купон (несколько ставок в экспресс)

```python
from bukmeker.coupon import ValueBetCandidate, generate_coupons

candidates = [
    ValueBetCandidate(bet_id=1, match_id=1001, league_id=1, team_ids=(1, 2), prob=0.55, odds=2.00),
    ValueBetCandidate(bet_id=2, match_id=1002, league_id=1, team_ids=(3, 4), prob=0.40, odds=2.85),
]
coupons = generate_coupons(candidates, bankroll=10_000, max_events=3, max_corr=0.3, top_n=5)
```

`max_corr` — порог допустимой корреляции между ногами купона (0.3 по
умолчанию, из исходной спецификации); ставки на один и тот же матч или одну
и ту же команду будут отфильтрованы автоматически.

## 7. Мультиспорт и сущности

```python
from bukmeker.entities import build_seed_registry

reg = build_seed_registry()
print(len(reg.countries))  # 249 — реальный полный список ISO 3166-1 (через pycountry)

premier_league = next(lg for lg in reg.leagues.values() if "Premier League" in lg.name)
clubs = reg.competitors_for_league(premier_league.id)  # Arsenal, Chelsea, ...

usa = reg.country_by_alpha3("USA")
reg.has_league_data(usa.id)  # True — MLS и NBA сидированы для США
```

Список стран — реальный и полный (все 249 официально признанных ISO-стран/
территорий). Лиги и клубы — курируемая, реальная (не вымышленная), но не
исчерпывающая подборка: ~35 футбольных стран с топ-лигой и несколькими
известными клубами, полный состав NBA (30 команд — единственный случай, где
"полный" в буквальном смысле), и известные ATP-игроки. Для подавляющего
большинства из 249 стран лиг/клубов в seed-реестре нет — это ожидаемо, не
баг (см. раздел 9, как подключить реальные данные через `--sync`). Баскетбол
и теннис используют свои модели, но тот же движок value-betting/Kelly/coupon:

```python
from bukmeker.sports.basketball import predict_moneyline
from bukmeker.sports.tennis import predict_match_win_prob

nba_probs = predict_moneyline(mu_margin=3.5, sigma_margin=12.0)          # {'home_win':..., 'away_win':...}
atp_probs = predict_match_win_prob(p_set=0.55, best_of=5)                 # {'player1_win':..., 'player2_win':...}
```

## 8. Монетизация купона

```python
from bukmeker.monetization import build_coupon_report, format_coupon_report

report = build_coupon_report(coupons[0], stake=100.0, platform_fee_pct=0.05)
print(format_coupon_report(report))
```

`platform_fee_pct` — ваша комиссия как процент от валовой выплаты
(`gross_payout`); поменяйте на 0.0, если комиссия не нужна.

## 9. Подключение реальных данных через AI-коннектор

Это единственная часть проекта, которая делает реальные сетевые запросы и
тратит реальные деньги (вызов Anthropic API оплачивается). Понадобится:

1. **Ключ провайдера спортивных данных** — любой REST/JSON API (например,
   API-Football, TheOddsAPI, Sportmonks и т.д.). Получите его на сайте
   провайдера.
2. **Ключ Anthropic API** — https://console.anthropic.com → API Keys.

### Через CLI

```bash
export SOURCE_API_KEY="ключ_вашего_провайдера"
export ANTHROPIC_API_KEY="ваш_ключ_anthropic"

bukmeker connector \
  --source-url https://api.ваш-провайдер.com \
  --path "fixtures?date=2026-07-05" \
  --key-location header \
  --key-name x-api-key
```

Флаги:
- `--source-url` — базовый адрес API провайдера (обязателен).
- `--path` — конкретный эндпоинт/запрос (по умолчанию `fixtures`).
- `--key-location header|query` — куда провайдер ожидает ключ: в заголовке
  запроса или как query-параметр (смотрите документацию провайдера).
- `--key-name` — имя заголовка/параметра для ключа (например,
  `x-apisports-key`, `apiKey` — тоже из документации провайдера).
- `--model` — модель Anthropic для маппинга полей (по умолчанию
  `claude-sonnet-5`).

Без ключей команда откажется работать и покажет подсказку по использованию —
это осознанное поведение, чтобы не притворяться, что данные реальны.

### Синхронизация в реестр сущностей (`--sync`)

Флаг `--sync` дополнительно сливает нормализованные записи в реестр сущностей
(`bukmeker.entities`) — так реестр наполняется данными реального провайдера,
а не только выводится в консоль:

```bash
bukmeker connector --source-url https://api.ваш-провайдер.com --path fixtures \
  --sync --sync-sport Football --sync-country GBR
```

Дополнительные флаги:
- `--sync-sport` — вид спорта (`Football`/`Basketball`/`Tennis`), к которому
  относятся новые лиги (по умолчанию `Football`).
- `--sync-country` — ISO alpha-3 код страны (например `GBR`, `USA`), к
  которой будут привязаны вновь обнаруженные лиги. Нужен потому, что
  провайдеры обычно не отдают страну лиги вместе с матчем — без этого флага
  честно определить страну для новой лиги нечем.

Повторный запуск `--sync` с теми же данными не создаёт дублей: лиги матчатся
по названию и виду спорта, команды — по названию внутри лиги.

### Программно

```python
from bukmeker.connectors import AIDataConnector, ClaudeFieldMapper, RawDataSource

source = RawDataSource(
    base_url="https://api.ваш-провайдер.com",
    api_key="ключ_вашего_провайдера",
    key_location="header",       # или "query"
    key_name="x-api-key",
)
mapper = ClaudeFieldMapper(api_key="ваш_ключ_anthropic")
connector = AIDataConnector(source, mapper)

matches = connector.fetch_and_normalize("fixtures")
for m in matches:
    print(m.home_team, "vs", m.away_team, m.home_odds, m.draw_odds, m.away_odds)
```

Как это работает внутри: коннектор один раз берёт пример записи из ответа
провайдера, показывает её Anthropic API и просит определить, в каком месте
JSON-структуры лежит каждое каноническое поле (`home_team`, `away_odds` и
т.д.). Дальше эта карта полей применяется ко всем записям без повторных
вызовов ИИ. Так поддерживается **любой** провайдер, даже незнакомый заранее.

## 10. Тестирование и разработка

```bash
pytest tests/ -q     # 130 тестов, без реальных сетевых/AI вызовов (используются фейки)
ruff check .           # линт, используется в CI
```

## 11. Частые проблемы

| Проблема | Причина / решение |
|---|---|
| `ValueError: odds must be > 1.0` | В `expected_value`/`kelly_stake` передан коэффициент ≤ 1 — проверьте входные данные. |
| `ValueError: odds imply no bookmaker margin` | В `shin_margin_removal` сумма implied-вероятностей ≤ 1 — такие коэффициенты не содержат маржи, снимать нечего. |
| Кракозябры вместо кириллицы в выводе на Windows | CLI сам переключает `stdout` в UTF-8 при старте (`bukmeker demo`/`connector`); если запускаете код иначе, добавьте `sys.stdout.reconfigure(encoding="utf-8")` перед выводом. |
| `bukmeker connector` печатает Usage и завершается с кодом 1 | Не заданы `SOURCE_API_KEY`/`ANTHROPIC_API_KEY` (или флаги `--source-key`/`--anthropic-key`) — это ожидаемое поведение, не ошибка. |
| `ModuleNotFoundError: anthropic` / `requests` | Установите с extras: `pip install -e ".[connectors]"`. |
| `ModuleNotFoundError: streamlit` / `bukmeker dashboard` пишет, что streamlit не установлен | Установите с extras: `pip install -e ".[dashboard]"`. |
| `bukmeker dashboard` не открывает браузер сам | Откройте вручную адрес из терминала (обычно `http://localhost:8501`). |
| Порт 8501 уже занят | Запустите на другом порту: `bukmeker dashboard --port 8600`. |
| Хочу добавить свою лигу/страну/клуб | Дополните `bukmeker/entities.py` (`build_seed_registry`) — это просто демонстрационные данные, структура не ограничивает список. Для реальных данных используйте `connectors/` вместо ручного добавления. |
