# MATH CORE PROMPT — Value Betting Quantitative Engine

Дистиллированная инженерная спецификация, выведенная из `bukmeker.txt`. Область —
математическое ядро value-betting платформы (без Node/React/RabbitMQ/K8s
инфраструктуры). Этот файл — не только исходный промт, но и актуальный протокол
принятых архитектурных решений: он обновляется по мере того, как ядро превращается
из набора формул в завершённый, пригодный к установке проект.

## Роль
Ты — квант-разработчик, реализующий детерминированное, воспроизводимое ядро оценки
вероятностей исходов футбольных матчей и обнаружения value bets, пригодное для
дальнейшей интеграции в промышленный пайплайн (Feature Store → Model → Calibration →
Margin Removal → Value Detection → Bankroll/Coupon).

## Обязательные математические компоненты

1. **Взвешивание истории**: экспоненциальное затухание `w(t) = e^{-λt}`, λ по умолчанию
   0.005 (полураспад ≈138 дней); скользящее взвешенное среднее и EMA.
2. **Рейтинги команд**:
   - Elo с домашним преимуществом `H`: `E = 1 / (1 + 10^{-(R_home - R_away + H)/400})`.
   - Байесовский рейтинг: гауссово убеждение `θ ~ N(μ0, σ0²)`, апостериорное обновление
     по каждому результату.
   - Poisson strength (атака/защита): `log(λ_home) = μ + home + attack_i - defence_j`,
     подгонка через максимизацию правдоподобия.
3. **Модели голов**: Poisson, бивариатный Poisson (ковариационный параметр λ3),
   Dixon-Coles (коррекция низких счётов через τ(0,0),τ(1,0),τ(0,1)), Negative Binomial
   (overdispersion), Skellam (разность голов), Monte-Carlo симуляция.
4. **Калибровка**: Platt scaling (логистическая регрессия поверх сырых оценок),
   изотоническая регрессия, Expected Calibration Error (ECE), Brier score, Log Loss.
5. **Удаление маржи**: метод Shin (решение по λ через `brentq`) и мультипликативный
   метод как baseline/fallback.
6. **Value Detection**: `EV = p·(odds-1) - (1-p)`, `Value% = p·odds - 1`,
   `Overlay = p_model - p_fair`, `Probability Edge = p_model/p_fair - 1`, bootstrap CI
   вероятности модели.
7. **Bankroll**: (Fractional) Kelly `f* = (p·(b-1) - (1-p)) / (b-1) = EV/(b-1)`, жёсткие
   лимиты (max stake %, max daily exposure).
8. **Coupon Generator**: перебор комбинаций до N событий, ограничение по корреляции
   исходов, максимизация ожидаемого логарифмического роста капитала (Half Kelly на
   совместную вероятность).

## Требования к качеству
- Без look-ahead и survivorship bias в интерфейсах (явные `as_of_date` там, где уместно).
- Каждая функция — чистая, детерминированная, типизированная (type hints), с docstring
  только там, где не очевиден смысл формулы.
- Численная устойчивость: защита от `odds <= 1`, `p` вне `[0,1]`, вырожденных матриц.
- Тестовое покрытие: unit-тесты на каждую математическую формулу с проверкой против
  независимого источника истины (scipy.stats, ручной расчёт, свойства суммы вероятностей).
- Итоговая демонстрация обязана прогонять полный путь: рейтинги → λ голов →
  Dixon-Coles матрица → 1X2 вероятности → Shin fair odds → EV/Value% → Kelly stake →
  генерация купона — на конкретном примере матча с печатью результатов.

## Критические ошибки, которых нужно избежать (см. bukmeker.txt §3.8)
Data/Look-Ahead Leakage, Overfitting, Selection/Survivorship Bias, Ignoring Margin,
Kelly Overbetting, Small Sample Bias, Correlation Risk, Closing Line Bias.

---

## Реализованные решения (журнал завершённого проекта)

Ниже — конкретные архитектурные и инженерные решения, принятые при реализации; они
фиксируют, *как именно* каждый пункт спецификации воплощён в коде, чтобы дальнейшие
итерации (например, при переходе к полной платформе) не переизобретали их заново.

### Структура пакета
```
bukmeker/                    # корень проекта (pip-устанавливаемый пакет)
├── pyproject.toml           # setuptools build, entry point `bukmeker`, extras [dev]
├── LICENSE                  # MIT
├── .gitignore
├── .github/workflows/ci.yml # GitHub Actions: ruff + pytest на 3.11 и 3.12
├── README.md
├── PROMPT.md                # этот файл
├── demo.py                  # тонкая обёртка над bukmeker.cli.run_demo()
├── bukmeker/                # импортируемый пакет
│   ├── features.py          # exponential_weight, weighted_rolling_mean, ema, home_advantage_score
│   ├── ratings.py            # elo_expected_score, elo_update, BayesianRating, PoissonStrength
│   ├── models/goals.py        # poisson_*, bivariate_poisson_matrix, dixon_coles_matrix,
│   │                          # negative_binomial_pmf, skellam_probs, monte_carlo_outcome_probs,
│   │                          # outcome_probs_from_matrix
│   ├── margin.py              # shin_margin_removal, multiplicative_margin_removal
│   ├── calibration.py         # PlattScaler, IsotonicScaler, expected_calibration_error, calculate_metrics
│   ├── value_betting.py       # expected_value, value_percentage, overlay, probability_edge,
│   │                          # bootstrap_probability_ci, kelly_stake, apply_global_limits
│   ├── coupon.py               # ValueBetCandidate, pairwise_correlation, combo_is_valid, generate_coupons
│   └── cli.py                  # argparse CLI: `bukmeker demo`
└── tests/                    # 55 unit-тестов, по одному файлу на модуль
```

### Ключевые инженерные решения и их обоснование
- **`PoissonStrength.fit`** оценивается через `scipy.optimize.minimize` (L-BFGS-B) с
  ridge-штрафом на attack/defence (0.01·Σθ²) для идентифицируемости параметров —
  без штрафа система вырождена (attack/defence можно сдвинуть на константу).
- **`dixon_coles_matrix`** клиппит отрицательные вероятности после τ-коррекции и
  ренормализует; предупреждение в docstring, что `rho` вне ~[-0.2, 0.2] не гарантирует
  валидности без клиппинга.
- **`bivariate_poisson_matrix`** проверен тестом на то, что при `lambda3=0` совпадает
  с независимым Poisson — это гарантирует, что общий (не частный) случай реализован
  правильно.
- **`shin_margin_removal`** явно бросает `ValueError`, если сумма implied-вероятностей
  ≤ 1 (нет маржи для снятия) — вместо тихого некорректного результата от `brentq`.
- **`outcome_probs_from_matrix`**: матрица индексируется `[home_goals, away_goals]`;
  home_win — нижний треугольник (`tril`, k=-1), away_win — верхний (`triu`, k=1).
  Перепутать их — самый вероятный источник silent bug, поэтому есть отдельный тест
  на вырожденной матрице с массой в одной ячейке.
- **`weighted_rolling_mean`** принимает `as_of_date` и фильтрует `df[date_col] < as_of_date`
  до агрегации — единственная точка, где предотвращается look-ahead в features.
- **`generate_coupons`** использует эвристику корреляции (`pairwise_correlation`):
  один матч → 1.0, общая команда → 0.6, одна лига → 0.25, иначе 0.0; порог по
  умолчанию 0.3 из документа. Это осознанное упрощение — реальная система должна
  заменить эвристику на исторически откалиброванную корреляционную матрицу исходов.
- **CLI**: `bukmeker demo` (после `pip install -e .`) и `python demo.py` (без
  установки) запускают идентичный сквозной прогон на synthetic-данных.
- **CI**: `ruff check .` + `pytest tests/ -q` на Python 3.11 и 3.12 при push/PR в `main`.

### Явные ограничения текущего объёма
Node.js backend, Prisma/PostgreSQL/TimescaleDB, RabbitMQ, React-фронтенд, Docker
Compose/Kubernetes и мониторинг (Prometheus/Grafana) из `bukmeker.txt` §1.2 сознательно
не реализованы — по согласованному решению текущий проект ограничен математическим
ядром как отдельной, тестируемой Python-библиотекой.
