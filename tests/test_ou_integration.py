"""
Validierung: unsere analytische MLE gegen Wergieluks numerische MLE.
Beide sollen auf denselben Daten ähnliche Ergebnisse liefern.
"""
import sys
import numpy as np
import pytest

# Wergieluk als optionale Referenz laden (falls verfügbar)
WERGIELUK_PATH = "/home/boobi/HAUPTLAGER/02_Research_Papers/wergieluk_julian/ou_noise"
HAS_WERGIELUK = False
try:
    sys.path.insert(0, WERGIELUK_PATH)
    from ou_noise import ou as wou
    HAS_WERGIELUK = True
except ImportError:
    pass

from quant_tools.ou.estimator import mle_analytical, half_life
from quant_tools.ou.simulator import path, noise_from_path
from quant_tools.ou.stationarity import is_stationary
from quant_tools.core.types import FitResult


def _simulate(a, mu, v, n=5000, seed=99):
    params = FitResult(
        mean_rev_speed=a, mean_rev_level=mu, vola=v,
        half_life=np.log(2)/a, loglik=0.0, n_obs=1
    )
    return path(x0=mu, n_steps=n, dt=1/252, params=params, seed=seed)


def test_full_pipeline_ou():
    """Simulieren → schätzen → stationär prüfen → half_life berechnen."""
    x = _simulate(a=3.0, mu=100.0, v=1.5)
    result = mle_analytical(x, dt=1/252)
    assert is_stationary(x)
    assert 1.0 < result.mean_rev_speed < 6.0
    assert 95.0 < result.mean_rev_level < 105.0
    assert result.half_life > 0
    # half_life sollte zwischen 20 und 250 Handelstagen liegen für a=3.0
    hl_days = result.half_life * 252
    assert 20 < hl_days < 250


@pytest.mark.skipif(not HAS_WERGIELUK, reason="Wergieluk-Code nicht verfügbar")
def test_compare_with_wergieluk():
    """Unsere analytische MLE vs. Wergieluks numerische MLE – max 15% Abweichung."""
    x = _simulate(a=2.0, mu=50.0, v=1.0, n=3000)
    dt = 1/252
    t = np.arange(len(x)) * dt

    our_result = mle_analytical(x, dt)
    werg_params = wou.mle(t, x)

    # mean_rev_speed
    assert abs(our_result.mean_rev_speed - werg_params[0]) / werg_params[0] < 0.15
    # mean_rev_level
    assert abs(our_result.mean_rev_level - werg_params[1]) / abs(werg_params[1]) < 0.05
