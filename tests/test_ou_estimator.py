import numpy as np
import pytest
from quant_tools.ou.estimator import mle_analytical, half_life


def test_half_life_basic():
    assert abs(half_life(1.0) - 0.6931471805599453) < 1e-10


def test_half_life_fast_reversion():
    assert half_life(10.0) < 0.1


def test_half_life_zero_speed_raises():
    with pytest.raises(ValueError):
        half_life(0.0)


def test_mle_analytical_recovers_params():
    """Simulierten OU-Pfad schätzen und Parameter zurückgewinnen."""
    rng = np.random.default_rng(42)
    a_true, mu_true, v_true = 2.0, 50.0, 1.0
    dt = 1.0 / 252
    n = 5000

    x = np.zeros(n)
    x[0] = mu_true
    exp_adt = np.exp(-a_true * dt)
    std = v_true * np.sqrt((1 - np.exp(-2 * a_true * dt)) / (2 * a_true))
    for i in range(1, n):
        x[i] = x[i - 1] * exp_adt + mu_true * (1 - exp_adt) + rng.normal(0, std)

    result = mle_analytical(x, dt)

    # Tolerance 15%: AR(1)-OLS ist bei phi≈1 (schnelle Mean-Reversion, kleines dt)
    # inhärent variabel. n=5000 reicht für ~15%, nicht für 10%.
    assert abs(result.mean_rev_speed - a_true) / a_true < 0.15
    assert abs(result.mean_rev_level - mu_true) / mu_true < 0.05
    assert abs(result.vola - v_true) / v_true < 0.15
    assert result.n_obs == n - 1


def test_mle_analytical_half_life_in_result():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 200)) + 100.0
    result = mle_analytical(x, dt=1.0)
    expected_hl = np.log(2) / result.mean_rev_speed
    assert abs(result.half_life - expected_hl) < 1e-10


def test_mle_analytical_min_length():
    with pytest.raises(ValueError, match="mindestens"):
        mle_analytical(np.array([1.0, 2.0]), dt=1.0)
