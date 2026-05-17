<p align="center">
  <h1 align="center">Quant Tools</h1>
  <p align="center">
    <strong>Statistical tooling for quantitative finance — Ornstein-Uhlenbeck processes, volatility estimation, and core types.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> · <a href="#ornstein-uhlenbeck">OU Module</a> · <a href="#validated-against">Validation</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/dependencies-numpy+scipy+pandas-orange" alt="Minimal deps">
  <img src="https://img.shields.io/badge/tests-6_suites_pass-brightgreen" alt="Tests passing">
</p>

---

## Why?

Most quant libraries are either too academic (no practical interface) or too black-box (no transparency into what's happening under the hood).

**Quant Tools is a middle ground** — correct math with clean Python APIs. Every estimator has the reference paper clearly documented, and results are validated against independent implementations.

Currently focused on:

- **Ornstein-Uhlenbeck processes** — MLE fitting, simulation, stationarity tests, bias correction
- **Volatility estimation** — quadratic variation, realized volatility, efficient estimators
- **Core types** — Pydantic models for `FitResult`, `SimParams`, `VolaResult` with serialization

---

## Quick Start

```bash
pip install quant-tools
```

Or from source:

```bash
git clone https://github.com/nessos666/quant-tools.git
cd quant-tools
pip install -e .
```

---

## Ornstein-Uhlenbeck

The OU module provides end-to-end workflow for mean-reverting processes.

### Fit an OU process to market data

```python
from quant_tools.ou.estimator import fit_mle, half_life, bias_correct
from quant_tools.core.types import FitResult

# prices = pd.Series(...)  # your price data
prices = ...

# MLE fit — discrete-time exact likelihood
result: FitResult = fit_mle(prices, dt=1/252)

print(f"Mean reversion speed θ: {result.mean_rev_speed:.4f}")
print(f"Mean reversion level μ:  {result.mean_rev_level:.4f}")
print(f"Volatility σ:            {result.vola:.4f}")
print(f"Half-life:               {half_life(result.mean_rev_speed):.1f} days")

# Apply finite-sample bias correction
corrected = bias_correct(result, dt=1/252)
```

### Simulate OU paths

```python
from quant_tools.ou.simulator import path
import numpy as np

# Simulate 10 years of daily data
sim = path(
    x0=0.0,
    n_steps=2520,
    dt=1/252,
    params=FitResult(mean_rev_speed=5.0, mean_rev_level=0.0, vola=0.02),
    seed=42
)
# sim is a numpy array of shape (2520,)
```

### Stationarity tests

```python
from quant_tools.ou.stationarity import adf_test, hurst_exponent

# Augmented Dickey-Fuller with OU-aware lag selection
adf_result = adf_test(prices)
print(f"ADF statistic: {adf_result.statistic:.4f}, p-value: {adf_result.pvalue:.4f}")

# Hurst exponent via rescaled range (R/S)
h = hurst_exponent(prices)
print(f"Hurst: {h:.4f}")  # < 0.5 = mean-reverting, > 0.5 = trending
```

### Volatility estimation

```python
from quant_tools.ou.estimator import est_vola_qv

# Model-free volatility from quadratic variation
vola = est_vola_qv(prices, dt=1/252)
print(f"Annualized volatility: {vola:.4f}")
```

---

## Validated Against

| Implementation | Result | Status |
|---------------|--------|--------|
| Wergieluk (2019) OU-MLE | θ, μ, σ match to 1e-6 | ✅ Verified |
| Analytical OU transition density | Closed-form vs. simulation | ✅ Verified |
| Scipy ADF test | p-values match | ✅ Verified |
| Hurst R/S (Weron 2002) | H within ±0.01 | ✅ Verified |
| Quadratic variation (Barndorff-Nielsen) | RV matches reference | ✅ Verified |

> See `tests/test_ou_integration.py` for the full cross-validation suite.

---

## Project Structure

```
quant_tools/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── types.py           # FitResult, SimParams, VolaResult (Pydantic)
└── ou/
    ├── __init__.py
    ├── estimator.py        # fit_mle, bias_correct, est_vola_qv, transition_density
    ├── simulator.py        # exact OU path simulation
    └── stationarity.py     # adf_test, hurst_exponent
tests/
├── test_core_types.py
├── test_ou_estimator.py
├── test_ou_integration.py  # Cross-validation against reference implementations
├── test_ou_rolling.py
├── test_ou_simulator.py
└── test_ou_stationarity.py
```

---

## Testing

```bash
pytest tests/ -v
```

All 6 test suites pass. The integration test validates against the Wergieluk reference implementation (included under `examples/ou_noise`).

---

## Philosophy

1. **Correctness first** — every estimator cross-validated against academic reference implementations
2. **Clean interfaces** — typed, documented, Pydantic-backed models
3. **No black boxes** — open math, clear references
4. **Minimal dependencies** — numpy, scipy, pandas, pydantic, loguru. No ML bloat.

---

## License

MIT — use it, validate it, improve it.

<p align="center">
  <small>Built for systematic NQ futures research.<br>
  <strong>github.com/nessos666</strong></small>
</p>
