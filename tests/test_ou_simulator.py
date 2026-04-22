import numpy as np
from quant_tools.ou.simulator import (
    noise_from_path,
    path,
    transition_logpdf,
    transition_pdf,
)
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
    assert abs(result.vola - 2.0) / 2.0 < 0.15


def test_transition_pdf_integrates_to_one():
    params = _make_params(a=2.0, mu=50.0, v=1.0)
    x_grid = np.linspace(40, 60, 10000)
    pdf_vals = np.array([transition_pdf(50.0, xi, 1 / 252, params) for xi in x_grid])
    dx = x_grid[1] - x_grid[0]
    integral = np.sum(pdf_vals) * dx
    assert abs(integral - 1.0) < 0.01


def test_transition_logpdf_consistent_with_pdf():
    params = _make_params(a=3.0, mu=100.0, v=2.0)
    pdf_val = transition_pdf(100.0, 101.0, 1 / 252, params)
    logpdf_val = transition_logpdf(100.0, 101.0, 1 / 252, params)
    assert abs(np.log(pdf_val) - logpdf_val) < 1e-10


def test_transition_pdf_peaks_at_conditional_mean():
    params = _make_params(a=5.0, mu=100.0, v=1.0)
    x0 = 110.0
    dt = 1 / 252
    exp_adt = np.exp(-5.0 * dt)
    cond_mean = x0 * exp_adt + 100.0 * (1 - exp_adt)

    pdf_at_mean = transition_pdf(x0, cond_mean, dt, params)
    pdf_away = transition_pdf(x0, cond_mean + 1.0, dt, params)
    assert pdf_at_mean > pdf_away


def test_transition_logpdf_returns_float():
    params = _make_params(a=2.0, mu=50.0, v=1.0)
    val = transition_logpdf(50.0, 50.1, 1 / 252, params)
    assert isinstance(val, float)
