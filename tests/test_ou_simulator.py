import numpy as np
from quant_tools.ou.simulator import path, noise_from_path
from quant_tools.ou.estimator import mle_analytical
from quant_tools.core.types import FitResult


def _make_params(a=2.0, mu=50.0, v=1.0) -> FitResult:
    return FitResult(
        mean_rev_speed=a,
        mean_rev_level=mu,
        vola=v,
        half_life=np.log(2) / a,
        loglik=0.0,
        n_obs=1,
    )


def test_path_length():
    params = _make_params()
    x = path(x0=50.0, n_steps=1000, dt=1 / 252, params=params, seed=0)
    assert len(x) == 1000


def test_path_starts_at_x0():
    params = _make_params()
    x = path(x0=42.0, n_steps=500, dt=1 / 252, params=params, seed=1)
    assert x[0] == 42.0


def test_path_mean_reversion():
    """Langer Pfad sollte um mu herum liegen."""
    params = _make_params(a=5.0, mu=100.0, v=0.5)
    x = path(x0=100.0, n_steps=10000, dt=1 / 252, params=params, seed=2)
    assert abs(x.mean() - 100.0) < 2.0


def test_noise_from_path_has_unit_variance():
    """Extrahierte Residuen sollen N(0,1) sein."""
    params = _make_params()
    x = path(x0=50.0, n_steps=5000, dt=1 / 252, params=params, seed=3)
    z = noise_from_path(x, dt=1 / 252, params=params)
    assert abs(z.mean()) < 0.1
    assert abs(z.std() - 1.0) < 0.1


def test_roundtrip_simulate_and_estimate():
    """Simulieren → schätzen → Parameter zurückgewinnen."""
    params = _make_params(a=3.0, mu=200.0, v=2.0)
    x = path(x0=200.0, n_steps=50000, dt=1 / 252, params=params, seed=42)
    result = mle_analytical(x, dt=1 / 252)
    # MLE hat bekannten downward bias beim mean_rev_speed – 30% Toleranz realistisch
    assert abs(result.mean_rev_speed - 3.0) / 3.0 < 0.30
    assert abs(result.mean_rev_level - 200.0) / 200.0 < 0.05
