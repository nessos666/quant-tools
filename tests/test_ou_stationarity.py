import numpy as np
import pytest
from quant_tools.ou.stationarity import adf_test, is_stationary
from quant_tools.ou.simulator import path
from quant_tools.core.types import FitResult


def _ou_params(a=5.0, mu=100.0, v=1.0):
    return FitResult(
        mean_rev_speed=a, mean_rev_level=mu, vola=v,
        half_life=np.log(2)/a, loglik=0.0, n_obs=1
    )


def test_adf_returns_pvalue():
    x = path(x0=100.0, n_steps=1000, dt=1/252, params=_ou_params(), seed=0)
    result = adf_test(x)
    assert "pvalue" in result
    assert "statistic" in result
    assert 0.0 <= result["pvalue"] <= 1.0


def test_stationary_process_detected():
    x = path(x0=100.0, n_steps=2000, dt=1/252, params=_ou_params(a=10.0), seed=1)
    assert is_stationary(x, alpha=0.05) is True


def test_random_walk_not_stationary():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0, 1, 2000))
    assert is_stationary(x, alpha=0.05) is False


def test_adf_min_length():
    with pytest.raises(ValueError, match="mindestens"):
        adf_test(np.array([1.0, 2.0, 3.0]))
