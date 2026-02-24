# CLAUDE.md — StochEstimate

## What this project is

**StochEstimate** is a thesis project comparing two approaches to estimating Ornstein-Uhlenbeck (OU) process parameters from financial spread time series:
- **MLE** (`estimation/mle.py`) — exact discrete likelihood, numerical Hessian for CIs
- **LSTM** (`estimation/lstm_estimator.py`) — trained on synthetic OU paths, Monte Carlo Dropout for uncertainty

The core thesis claim is: *LSTM-based estimation can match or approximate MLE accuracy on OU parameter recovery while providing richer uncertainty quantification through MC Dropout.*

Everything in this codebase serves that comparison. Do not add features that don't serve it.

---

## Workflow rules (before writing any code)

### 1. Brainstorm before coding
For any non-trivial task, explore before implementing. Ask:
- What already exists that I could reuse or extend?
- What are the 2–3 reasonable approaches?
- What are the failure modes?
- Does this align with the thesis comparison goal?

### 2. Write a plan, get approval
For tasks touching more than one file or introducing new logic:
- State exactly which files change and why
- Describe the approach in plain English
- Note any risks or open questions
- Use `EnterPlanMode` and wait for explicit approval before writing code

### 3. Execute in phases with checkpoints
Break execution into logical phases (e.g., data → model → integration → test).
After each phase: verify outputs, run the relevant test, confirm correctness before moving on.

### 4. Verify before declaring done
Never say a task is complete without evidence:
- For new functions: run them on a known input and show output
- For pipeline changes: run `test_lstm_pipeline.py` or the relevant pipeline path
- For DB changes: confirm the insert/query round-trips correctly
- For numerical code: spot-check against a synthetic case with known answer (e.g., θ=0.05, μ=0, σ=1)

### 5. Do not bypass validation
The validation step (`validation/validation_framework.py`) is the gating condition before estimation. A spread that hasn't passed at least stationarity + autocorrelation tests must not be fed to MLE or LSTM. This is a non-negotiable architectural rule.

---

## Architecture overview

```
Data (Yahoo Finance)
    ↓
preprocessing/
  engle_granger_cointegration.py   — Engle-Granger 2-step test, produces spread
  validate_real_pairs.py           — Fetches prices, runs cointegration, returns spread
    ↓
validation/
  validation_framework.py          — Orchestrates 5 tests, computes confidence level
  stationarity_test.py             — ADF test
  linear_drift_test.py             — OLS slope significance
  constant_volatility.py           — Levene's test on split halves
  acf_exponential.py               — Exponential ACF decay fit (R² > 0.85)
  normality_test.py                — Shapiro-Wilk (informational only)
    ↓
estimation/
  mle.py                           — Nelder-Mead + Hessian CIs → OUEstimationResult
  lstm_estimator.py                — LSTM + MC Dropout → OULSTMResult
  train_lstm.py                    — Standalone training script
  backtesting.py                   — Pairs trading strategy, out-of-sample only
    ↓
pipeline.py                        — Orchestrates all phases, prints comparison table
database/db_manager.py             — PostgreSQL CRUD, connection pool
```

### Key data flow invariant
Training split is always **70% estimation / 30% backtesting** (`split_idx = int(n_total * 0.7)`). The backtest never sees the data used for parameter estimation. Do not change this without explicit instruction.

---

## Result object interfaces

These two classes must stay interface-compatible for drop-in comparison:

```python
# estimation/mle.py
@dataclass
class OUEstimationResult:
    theta: float
    mu: float
    sigma: float
    log_likelihood: float
    success: bool
    message: str
    theta_ci: tuple   # (lower, upper) 95%
    mu_ci: tuple
    sigma_ci: tuple
    detailed_result: str  # property

# estimation/lstm_estimator.py
@dataclass
class OULSTMResult:
    theta: float
    mu: float
    sigma: float
    theta_ci: tuple
    mu_ci: tuple
    sigma_ci: tuple
    theta_std: float
    mu_std: float
    sigma_std: float
    mc_draws: dict        # {'theta': [...], 'mu': [...], 'sigma': [...]}
    n_mc_samples: int
    detailed_result: str  # property
```

When adding fields to either, check the other doesn't break. The comparison table in `pipeline._print_estimation_comparison()` uses `.theta`, `.mu`, `.sigma`, `.theta_ci`, `.mu_ci`, `.sigma_ci` from both.

---

## Database schema

PostgreSQL 16, running in Docker on `localhost:5432` (`stochestimate` / `stochestimate_dev`).

```
pairs               — ticker1, ticker2, pair_name, sector
price_data          — pair_id → prices with OHLCV, interval, timestamp
cointegration_results — pair_id → hedge_ratio, adf_statistic, adf_pvalue, cointegrated
validation_results  — pair_id → 5 test results, confidence_level, tests_passed_count
lstm_models         — model metadata: filename, training stats, mae_validation_loss, version
```

All inserts go through `db_manager.py`. Never write raw SQL outside that file. All queries use parameterized statements — no string formatting in SQL.

Confidence levels: `'HIGH'` (4/4 tests), `'MEDIUM'` (3/4), `'LOW'` (2/4), `'NOT_OU'` (fails stationarity or ACF).

---

## Module conventions

### Naming
- Statistical test functions: `test_{name}(series, critical_level=0.05)` — returns a dataclass
- Test result dataclasses: `{Name}TestResult` with `.passed`, `.p_value`, `.detailed_result`
- Estimation result classes: `OU{Method}Result`
- DB insert methods: `insert_{table_name}()`
- DB query methods: `get_{table_name}(filters...)`

### Result objects
Every statistical test returns a dataclass (not a dict, not a tuple). Every dataclass has:
- `passed: bool` — the binary verdict
- Relevant numeric fields (statistic, p_value, etc.)
- `detailed_result` property or similar for human-readable output

### Print feedback
Use status symbols for visual clarity in printed output:
- `✓` — success / passed
- `✗` — failure / did not converge
- `⚠` — warning / borderline
- No emojis beyond these

### Error handling
- MLE: if Hessian is singular, fall back to rough scale estimates (don't crash)
- LSTM: if window is shorter than `window_size`, pad with first value on the left
- Pipeline: `try/except` per pair — one failing pair doesn't abort the full run
- DB: `execute_insert` returns `None` on failure — callers must handle `None`

---

## OU process mathematics

The model: `dX_t = θ(μ - X_t)dt + σ dW_t`

**Parameters:**
- `θ` (theta): mean reversion speed (per day). Half-life = ln(2)/θ days. Typical range 0.005–0.5.
- `μ` (mu): long-term equilibrium. For a normalized spread, close to 0.
- `σ` (sigma): instantaneous volatility. Always positive.

**Discrete likelihood (used in MLE):**
```
X_{t+dt} | X_t ~ N(m_t, v_t)
m_t = μ + (X_t - μ)·exp(-θ·dt)
v_t = σ²/(2θ) · (1 - exp(-2θ·dt))
```

**LSTM normalization:**
```
Input:  X_norm = (X - mean(X)) / std(X)
Output de-normalization:
  θ_real = θ            (scale-invariant)
  μ_real = mean + μ_norm · std
  σ_real = σ_norm · std
Predictions are in log-space: model outputs (log_θ, μ_norm, log_σ_norm)
```

**Validation ACF test:** fits `ACF(k) = exp(-θ·k)`, requires R² > 0.85 and 0.01 < θ < 2.0.

---

## Key commands

```bash
# Run the full pipeline (Phase 1 discovery + Phase 2 analysis)
python pipeline.py

# Run pipeline with LSTM comparison
# (modify main() to pass lstm_model_path='estimation/saved_models/ou_lstm_v1.pt')
python pipeline.py

# Train LSTM (thesis-quality: 50k samples, 100 epochs, ~40 min on CPU)
python estimation/train_lstm.py

# Integration smoke test (uses DB-cached pairs, loads existing model)
python test_lstm_pipeline.py

# Initialise DB schema (one-time)
python init_schema.py

# Start DB + pgAdmin
docker-compose up -d

# Run Phase 2 only on DB-cached cointegrated pairs
python pipeline_analysis.py
```

---

## What not to do

- **Don't add new pipeline phases** without updating `pipeline.py`, `pipeline_analysis.py`, and the DB schema together
- **Don't skip the 70/30 split** — using full-sample parameters for backtesting is look-ahead bias
- **Don't store model weights in git** for large models (>50 MB) — the current `ou_lstm_v1.pt` is 211 KB (fine)
- **Don't create new result dataclass shapes** — extend `OUEstimationResult` or `OULSTMResult` instead
- **Don't add thesis-unrelated features** — this is a focused comparison study, not a trading platform
- **Don't commit `.idea/`, `__pycache__/`, or `*.pyc`** — add a `.gitignore` if needed
- **Don't run pipeline with `skip_cached=False`** unless you've cleaned the DB first — it creates duplicate pairs

---

## Thesis scope boundaries

**In scope:**
- OU parameter estimation accuracy: MLE vs LSTM (θ, μ, σ)
- Uncertainty quantification: Fisher Information (MLE) vs MC Dropout (LSTM)
- Out-of-sample validation through backtesting
- Synthetic benchmark (known parameters) + real financial pairs

**Out of scope:**
- Real-time trading infrastructure
- Other stochastic processes (Heston, jump-diffusion, etc.)
- Intraday data (thesis uses `1d` interval only)
- Portfolio optimization across pairs
- Alternative deep learning architectures (Transformer, TCN, etc.) unless specifically requested

---

## Project file map

```
StochEstimate/
├── pipeline.py                        # Main orchestrator (3-phase hybrid)
├── pipeline_analysis.py               # Phase 2-only, loads pairs from DB
├── init_schema.py                     # One-time DB schema creation
├── test_lstm_pipeline.py              # Integration smoke test
├── docker-compose.yml                 # PostgreSQL + pgAdmin
│
├── preprocessing/
│   ├── engle_granger_cointegration.py # Cointegration test
│   └── validate_real_pairs.py         # Yahoo Finance fetcher + cointegration
│
├── validation/
│   ├── validation_framework.py        # Master orchestrator (ValidationReport)
│   ├── stationarity_test.py
│   ├── linear_drift_test.py
│   ├── constant_volatility.py
│   ├── acf_exponential.py
│   └── normality_test.py
│
├── estimation/
│   ├── mle.py                         # OUEstimationResult, estimate_ou_mle()
│   ├── lstm_estimator.py              # OULSTMResult, OULSTMEstimator
│   ├── train_lstm.py                  # Standalone training + evaluation
│   ├── backtesting.py                 # BacktestResult, backtest_pairs_trading()
│   └── saved_models/
│       └── ou_lstm_v1.pt              # Trained model (20k samples, 100 epochs)
│
├── synthetic_data/
│   └── ou_generator.py                # generate_ou_process(), generate_random_walk()
│
├── database/
│   └── db_manager.py                  # DatabaseManager, all CRUD methods
│
└── visualization/
    └── validation_visualization.py    # Diagnostic plots
```
