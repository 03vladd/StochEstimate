# Research Progress — StochEstimate Thesis

**Last updated:** 2026-02-24
**Status:** Experiments complete, writing phase not yet started

---

## Thesis in one sentence

> We show that an LSTM trained on synthetic OU paths estimates (θ, σ) with accuracy
> comparable to classical MLE on clean data, and significantly outperforms MLE when
> real-world jump contamination is present — degrading ~3× more slowly in σ error
> across all tested pair types.

---

## What has been built

### Module map

```
StochEstimate/
│
├── preprocessing/          Data fetching and spread construction
│   └── data_fetcher.py       yfinance → spread = close_t1 - β·close_t2
│
├── estimation/             Parameter estimation
│   ├── mle.py                MLE via exact discrete OU likelihood (Nelder-Mead)
│   ├── lstm_estimator.py     LSTM + MC Dropout estimator (trained on synthetic data)
│   ├── train_lstm.py         Standalone training script
│   └── robustness_experiment.py  Main experimental contribution
│
├── validation/             OU-likeness testing
│   └── ou_validator.py       5-test battery returning HIGH/MEDIUM/LOW/NOT_OU
│
├── synthetic_data/         Synthetic process generation
│   └── ou_generator.py       generate_ou_process() + add_jump_contamination()
│
├── backtesting/            Pairs trading signal evaluation
│   └── backtester.py         Z-score entry/exit, P&L, Sharpe ratio
│
├── database/               PostgreSQL persistence
│   └── db_manager.py         CRUD for pairs, cointegration, validation, price, LSTM metadata
│
├── pipeline.py             End-to-end orchestrator (Phase 1 + Phase 2)
│
└── estimation/saved_models/
    └── ou_lstm_v1.pt         Trained LSTM weights (50k samples, 100 epochs)
```

---

## Phase 1 — Pair Selection (DONE)

**Goal:** Screen a universe of stock pairs and identify those whose spread is cointegrated.

**Method:**
- Engle-Granger cointegration test on the hedge-ratio-adjusted spread
- ADF test on residuals; cointegrated if p < 0.05
- Pairs tested: subset of S&P 500 stocks grouped by sector

**Results stored in DB (table: `cointegration_results`):**

| Pair | ADF stat | p-value | Hedge ratio |
|---|---|---|---|
| Morgan Stanley / Goldman Sachs | -2.879 | 0.048 | ~1.0 |
| PPL / Ameren | -4.264 | 0.001 | ~1.0 |
| EOG Resources / Schlumberger | -3.119 | 0.025 | ~1.0 |
| Amazon / Walmart | -3.581 | 0.006 | ~1.0 |
| Exelon / AEP | -3.039 | 0.031 | ~1.0 |

**5 cointegrated pairs found** across the tested universe.

---

## Phase 2 — OU Validation (DONE)

**Goal:** Of the cointegrated pairs, determine which most closely follow OU dynamics (not just stationary, but with the right structure).

**5-test battery** (each test passes/fails, final label is HIGH/MEDIUM/LOW/NOT_OU):

| Test | What it checks | Tool |
|---|---|---|
| Stationarity | ADF on spread residuals | statsmodels |
| No drift | Spread has no persistent trend | t-test on first differences |
| Volatility stability | σ doesn't change over time | Levene's test (rolling windows) |
| Autocorrelation | Mean-reversion structure exists | Ljung-Box test |
| Normality | Residuals are approximately Gaussian | Shapiro-Wilk |

**Confidence labels:**

| Pair | Level | Tests passed |
|---|---|---|
| Morgan Stanley / Goldman Sachs | **HIGH** | 4/5 |
| EOG Resources / Schlumberger | MEDIUM | 3/5 |
| Amazon / Walmart | MEDIUM | 3/5 |
| Exelon / AEP | MEDIUM | 3/5 |
| PPL / Ameren | LOW | 2/5 |

**Key observation:** Confidence level does not map to mean-reversion speed (θ).
All pairs have θ ≈ 0.03–0.05 (half-life 14–25 days). What differs is noise structure:
HIGH-confidence pairs have more Gaussian, more stable noise; LOW-confidence pairs
have heavier tails and less stable volatility.

---

## Phase 3 — Parameter Estimation Comparison (DONE)

### Estimator 1: MLE

- Exact discrete OU likelihood: `X_{t+dt} | X_t ~ N(μ + (X_t − μ)e^{−θΔt}, σ²/2θ · (1 − e^{−2θΔt}))`
- Optimizer: Nelder-Mead (scipy)
- Confidence intervals: numerical Hessian → Fisher information matrix
- Theoretically optimal on clean Gaussian OU data (Cramér-Rao lower bound)

### Estimator 2: LSTM with Monte Carlo Dropout

- Architecture: LSTM(input=1, hidden=64, layers=2, dropout=0.2) → FC(64→32) → FC(32→3)
- Outputs: `(log θ, μ_norm, log σ_norm)` in normalized space
- Training: 50,000 synthetic OU paths, 100 epochs, Adam + ReduceLROnPlateau
  - θ ~ LogUniform(0.005, 0.5), μ = 0 (identifiable only in normalized space), σ ~ Uniform(0.1, 2.0)
- Uncertainty: 200 MC Dropout forward passes at inference → 95% CI from percentiles
- De-normalization: θ is scale-invariant; μ_real = w_mean + μ_norm·w_std; σ_real = σ_norm·w_std

**Clean data performance** (1000-path synthetic test set, θ=0.05, σ=1.0):

| Metric | MLE | LSTM |
|---|---|---|
| θ MAE | ~0.028 | 0.029 |
| σ MAE | ~0.047 | 0.054 |

Both methods are comparable. MLE has a slight edge (expected — Cramér-Rao).

**Known limitation — μ estimation:**
Both methods have high μ MAE (~2.0 for LSTM on normalized scale). This is a fundamental
identifiability problem: for slow-reverting processes (θ ≈ 0.03–0.05), the stationary
variance σ²/2θ is large, so the sample mean over a finite window has high variance.
Neither method can reliably estimate μ on short windows. **μ is excluded from the
thesis comparison; the focus is θ and σ.**

---

## Phase 4 — Robustness Experiment (DONE)

**Core question:** When the OU assumption is violated (real-world contamination),
which estimator degrades more gracefully?

### Scenario A: Jump Outlier Contamination

A fraction of observations is replaced with spikes drawn from `N(0, 5σ)`.
This models earnings announcements, macro shocks, flash crashes.

**Results (200 synthetic paths per contamination level):**

| Jump % | MLE σ MAE | LSTM σ MAE | LSTM advantage |
|---|---|---|---|
| 0% (clean) | 0.047 | 0.054 | MLE slightly better (baseline) |
| 2% | 2.030 | 0.600 | LSTM **3.4× better** |
| 5% | 4.635 | 1.525 | LSTM **3.0× better** |
| 10% | 8.741 | 2.879 | LSTM **3.0× better** |

**Why MLE breaks:** The Gaussian log-likelihood penalizes large residuals quadratically.
A single spike `(X_{i+1} − m_i)²/v_i` can be enormous, dominating the loss and pulling
σ̂ upward dramatically. LSTM, with no parametric noise assumption, suppresses isolated
anomalies via gating and degrades much more slowly.

### Stratified by Validation Confidence Profile

The same experiment, repeated using OU parameters matching each confidence level
(derived from actual pair MLE estimates):

| Profile | True θ | True σ | Example pair |
|---|---|---|---|
| HIGH | 0.034 | 1.02 | Morgan Stanley / Goldman Sachs |
| MEDIUM | 0.035 | 2.00 | EOG/SLB, AMZN/WMT, EXC/AEP (avg) |
| LOW | 0.054 | 0.22 | PPL / Ameren |

**Summary table (σ MAE / σ_true — normalized for comparability, 200 paths):**

|  | 0% (clean) | | 2% jumps | | 5% jumps | | 10% jumps | |
|---|---|---|---|---|---|---|---|---|
| Profile | MLE | LSTM | MLE | LSTM | MLE | LSTM | MLE | LSTM |
| HIGH | 0.043 | 0.056 | 2.375 | 0.692 | 5.362 | 1.779 | 10.000 | 3.377 |
| MEDIUM | 0.043 | 0.055 | 2.349 | 0.686 | 5.308 | 1.760 | 9.905 | 3.338 |
| LOW | 0.048 | 0.053 | 1.963 | 0.579 | 4.492 | 1.470 | 8.509 | 2.783 |

**LSTM wins 9/9 contaminated comparisons (all profiles × all non-zero jump rates).**

**Conclusion:** The LSTM robustness advantage is not a fluke of one parameter regime.
It holds regardless of validation confidence level. This validates the two-stage design:
filter for OU-likeness first (Phase 2), then use LSTM estimation (Phase 3/4).

---

## Thesis narrative arc

```
1. Motivation
   Pairs trading requires accurate OU parameter estimates.
   Real financial spreads are not perfectly Gaussian — jumps happen.

2. Baseline (Chapter: MLE)
   MLE is the gold standard on clean OU data (Cramér-Rao optimal).
   Implement and validate on synthetic + real pairs.

3. LSTM estimator (Chapter: LSTM)
   Train on synthetic data → amortized inference (ms vs seconds for MLE).
   Comparable accuracy on clean data, uncertainty via MC Dropout.

4. Validation framework (Chapter: Pair selection)
   5-test OU-likeness battery → HIGH/MEDIUM/LOW confidence.
   Real pairs: 1 HIGH, 3 MEDIUM, 1 LOW across tested universe.

5. Robustness experiment (Chapter: Main contribution)
   LSTM degrades ~3× more slowly than MLE under jump contamination.
   Advantage is consistent across all validation confidence profiles.
   → LSTM is preferable in practice because real data always has jumps.

6. Conclusion
   Two-stage approach: validate OU-likeness first, then prefer LSTM estimation.
   μ estimation remains hard for both methods; focus on θ and σ.
   Future work: train on jump-augmented data, test more contamination scenarios.
```

---

## Key numbers to cite in the thesis

| Fact | Value |
|---|---|
| LSTM training data | 50,000 synthetic OU paths, 100 epochs |
| LSTM architecture | 2-layer LSTM (hidden=64), FC(64→32→3), dropout=0.2 |
| MC Dropout samples | 200 per inference call |
| Final val loss (MSE on log-space targets) | 0.471 |
| θ MAE on test set (LSTM) | 0.029 |
| σ MAE on test set (LSTM) | 0.054 |
| Pairs tested for cointegration | ~50 (sector-grouped) |
| Cointegrated pairs found | 5 |
| HIGH confidence pairs | 1 (Morgan Stanley / Goldman Sachs) |
| LSTM robustness advantage at 10% jump rate | ~3.0× lower σ MAE vs MLE |
| LSTM wins in stratified comparison | 9 / 9 |

---

## What remains

| Task | Priority | Notes |
|---|---|---|
| Visualization: training loss curves | High | Thesis figure: epoch vs train/val loss |
| Visualization: robustness degradation curves | High | Thesis figure: jump rate vs MAE for both methods |
| Visualization: MC Dropout CI comparison (MLE vs LSTM on real pair) | High | Shows uncertainty quantification side-by-side |
| Write Chapter: MLE | High | Mostly the math from mle.py docstring |
| Write Chapter: LSTM estimator | High | Architecture, training, MC Dropout, normalization strategy |
| Write Chapter: Validation framework | Medium | 5-test battery, confidence levels, real pair results |
| Write Chapter: Robustness experiment | Medium | Main experimental contribution |
| Scenario B: Model misspecification | Optional | Test on AR(1) or GARCH paths instead of OU |
| Retrain LSTM with jump augmentation | Optional | Would close the gap at high contamination levels |
| More pairs / longer history | Optional | Current: 5 pairs, 2 years — limited sample |

---

## Commands reference

```bash
# Train LSTM (full — ~40 min)
python -u estimation/train_lstm.py

# Run robustness experiment (base)
python estimation/robustness_experiment.py --n-paths 200

# Run stratified experiment
python estimation/robustness_experiment.py --stratified --n-paths 200

# Full pipeline smoke test (uses existing DB pairs)
python test_lstm_pipeline.py

# Full pipeline Phase 1+2 (downloads fresh data)
python pipeline.py
```
