from typing import Any

import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_test(x: np.ndarray, maxlag: int | None = None) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        raise ValueError(f"x braucht mindestens 20 Werte für ADF-Test, got {len(x)}")
    stat, pvalue, n_lags, n_obs, crit_vals, _ = adfuller(x, maxlag=maxlag, autolag="AIC")
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "n_lags": int(n_lags),
        "n_obs": int(n_obs),
        "critical_values": crit_vals,
    }


def is_stationary(x: np.ndarray, alpha: float = 0.05) -> bool:
    result = adf_test(x)
    return result["pvalue"] < alpha
