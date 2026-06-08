# StochEstimate

Codebase for the diploma thesis **"Robust Ornstein-Uhlenbeck Parameter Estimation via Neural Amortized Inference: A Five-Way Comparison with Classical Maximum Likelihood Methods"** — Babeș-Bolyai University, Faculty of Mathematics and Computer Science, 2026.

The repository contains the full experimental pipeline (five estimators, robustness benchmark, bootstrap significance testing), a split-conformal uncertainty calibration wrapper, and a prototype web application for interactive OU analysis.

---

## Repository Structure

```
StochEstimate/
│
├── estimation/                        # Core estimation methods
│   ├── mle.py                         # Gaussian MLE (Nelder-Mead + Hessian CIs)
│   ├── mle_robust.py                  # Student-t MLE (fixed df=4) + adaptive t-MLE
│   ├── lstm_estimator.py              # LSTM + MC Dropout (OULSTMResult)
│   ├── conformal_estimator.py         # Split-conformal calibration wrapper
│   ├── train_lstm.py                  # Train LSTM-v1 on clean synthetic paths
│   ├── train_lstm_robust.py           # Train LSTM-robust (50/50 clean/contaminated)
│   ├── robustness_experiment.py       # Main benchmark: 5 estimators × 4 jump levels
│   ├── calibration_check.py           # CI coverage experiments (MC Dropout + conformal)
│   ├── backtesting.py                 # Out-of-sample pairs trading backtest
│   ├── robustness_results_jumps.csv   # MAE results (θ, σ) across contamination levels
│   ├── calibration_coverage_conformal.csv  # Coverage table (MLE / MC / conformal)
│   └── saved_models/
│       ├── ou_lstm_v1.pt              # LSTM trained on clean paths (211 KB)
│       └── ou_lstm_v2_robust.pt       # LSTM-robust (50/50 training, 211 KB)
│
├── validation/                        # OU assumption validation battery
│   ├── validation_framework.py        # Orchestrator → ValidationReport
│   ├── stationarity_test.py           # ADF test
│   ├── linear_drift_test.py           # OLS slope significance
│   ├── constant_volatility.py         # Levene's test
│   ├── acf_exponential.py             # Exponential ACF decay (R² > 0.85)
│   └── normality_test.py              # Shapiro-Wilk (informational)
│
├── preprocessing/                     # Data acquisition and cointegration
│   ├── engle_granger_cointegration.py # Engle-Granger 2-step test
│   └── validate_real_pairs.py         # Yahoo Finance fetcher + cointegration
│
├── synthetic_data/
│   └── ou_generator.py                # generate_ou_process(), jump contamination
│
├── screening/
│   ├── screen_high_confidence_pairs.py  # Batch pair screening
│   └── screening_results.csv
│
├── visualization/
│   ├── robustness_visualization.py    # Degradation plots, frontier figure
│   └── validation_visualization.py   # Diagnostic plots for validation tests
│
├── database/
│   └── db_manager.py                  # PostgreSQL CRUD (all queries parameterized)
│
├── webapp/
│   ├── backend/                       # FastAPI + PostgreSQL
│   │   ├── app/
│   │   │   ├── api/v1/                # REST endpoints (auth, pairs, analysis)
│   │   │   ├── models/                # SQLAlchemy ORM models
│   │   │   ├── schemas/               # Pydantic schemas
│   │   │   └── services/              # Business logic (analysis, narration)
│   │   ├── alembic/                   # Database migrations
│   │   ├── requirements.txt
│   │   └── .env.example
│   └── frontend/                      # React + Vite + Tailwind + shadcn/ui
│       ├── src/
│       │   ├── pages/                 # Login, Register, Dashboard, PairDetail
│       │   ├── components/            # ParameterCard, ZScoreChart, ValidationPanel
│       │   └── api/                   # Typed API client
│       └── package.json
│
├── pipeline.py                        # Full pipeline: discovery → validation → estimation
├── pipeline_analysis.py               # Phase 2 only (loads pairs from DB)
├── init_schema.py                     # One-time DB schema creation
├── docker-compose.yml                 # PostgreSQL 16 + pgAdmin
└── test_lstm_pipeline.py              # Integration smoke test
```

---

## Quickstart — Research Pipeline

**Requirements:** Python 3.10+, Docker (for the database)

```bash
# 1. Install dependencies
pip install -r requirements.txt   # or use pyproject.toml

# 2. Start the database
docker-compose up -d

# 3. Initialise schema (once)
python init_schema.py

# 4. Run the full pipeline on real equity pairs
python pipeline.py

# 5. Re-run analysis only (uses DB-cached pairs)
python pipeline_analysis.py
```

---

## Robustness Benchmark

Reproduces the five-way comparison from Chapter 4 of the thesis.

```bash
# Run benchmark: 5 estimators × 4 contamination levels × 200 paths
python estimation/robustness_experiment.py

# Check CI coverage (MC Dropout vs split-conformal)
python estimation/calibration_check.py --conformal
```

Results are written to `estimation/robustness_results_jumps.csv` and `estimation/calibration_coverage_conformal.csv`.

---

## Training the LSTM

```bash
# Train LSTM-v1 on clean paths (~40 min on CPU, 50k paths, 100 epochs)
python estimation/train_lstm.py

# Train LSTM-robust on 50/50 clean/contaminated mix
python estimation/train_lstm_robust.py
```

Pre-trained weights are included in `estimation/saved_models/` and can be used directly without retraining.

---

## Webapp (Prototype Application)

The web application from Chapter 5 of the thesis. Requires the database to be running.

**Backend (FastAPI)**

```bash
cd webapp/backend
cp .env.example .env          # fill in DB credentials and Anthropic API key
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

**Frontend (React + Vite)**

```bash
cd webapp/frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173` with the API at `http://localhost:8000`.

---

## Key Parameters (Thesis Benchmark)

| Parameter | Value |
|-----------|-------|
| True θ* | 0.05 (half-life ≈ 13.9 days) |
| True μ* | 0.0 |
| True σ* | 1.0 |
| Path length | 200 observations |
| Paths per level | N = 200 |
| Contamination levels ε | 0%, 2%, 5%, 10% |
| Jump scale | 5 × σ̂_series |
| Bootstrap resamples | B = 2000 |
| MC Dropout passes | K = 200 |
| Conformal calibration set | 500 clean held-out paths |