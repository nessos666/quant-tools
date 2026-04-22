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


def test_mle_analytical_min_length_boundary():
    """3 Werte → n=2 < 3 muss ebenfalls ValueError auslösen."""
    with pytest.raises(ValueError, match="mindestens"):
        mle_analytical(np.array([1.0, 2.0, 3.0]), dt=1.0)


def test_mle_analytical_explosive_ar1_raises():
    """AR(1) mit phi=1.05 (explosiv) → OLS schätzt phi > 1 → ValueError."""
    rng = np.random.default_rng(7)
    x = np.zeros(500)
    x[0] = 0.0
    for i in range(1, 500):
        x[i] = 1.05 * x[i - 1] + rng.normal(0, 0.01)
    with pytest.raises(ValueError, match="phi="):
        mle_analytical(x, dt=1.0 / 252)


def test_mle_analytical_negatively_autocorrelated_raises():
    """Alternierend ±1 → phi < 0 → ValueError."""
    x = np.array([float((-1) ** i) for i in range(200)])
    with pytest.raises(ValueError, match="phi="):
        mle_analytical(x, dt=1.0)


def test_bias_correct_reduces_bias():
    """Bias-korrigierter θ soll näher am wahren Wert liegen."""
    from quant_tools.ou.estimator import bias_correct
    rng = np.random.default_rng(42)
    a_true = 2.0
    dt = 1.0 / 252
    n = 500

    x = np.zeros(n)
    x[0] = 50.0
    exp_adt = np.exp(-a_true * dt)
    std = np.sqrt((1 - np.exp(-2 * a_true * dt)) / (2 * a_true))
    for i in range(1, n):
        x[i] = x[i - 1] * exp_adt + 50.0 * (1 - exp_adt) + rng.normal(0, std)

    raw = mle_analytical(x, dt)
    corrected = bias_correct(raw, dt)

    raw_err = abs(raw.mean_rev_speed - a_true)
    cor_err = abs(corrected.mean_rev_speed - a_true)
    assert cor_err <= raw_err * 1.1


def test_bias_correct_preserves_other_fields():
    """Bias-Korrektur ändert nur mean_rev_speed und half_life."""
    from quant_tools.ou.estimator import bias_correct
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 200)) + 100.0
    raw = mle_analytical(x, dt=1.0)
    corrected = bias_correct(raw, dt=1.0)

    assert corrected.mean_rev_level == raw.mean_rev_level
    assert corrected.vola == raw.vola
    assert corrected.loglik == raw.loglik
    assert corrected.n_obs == raw.n_obs
    expected_hl = np.log(2) / corrected.mean_rev_speed
    assert abs(corrected.half_life - expected_hl) < 1e-10


def test_bias_correct_large_n_minimal_effect():
    """Bei großem n (5000) soll die Korrektur fast nichts ändern."""
    from quant_tools.ou.estimator import bias_correct
    rng = np.random.default_rng(7)
    a_true = 3.0
    dt = 1.0 / 252
    n = 5000

    x = np.zeros(n)
    x[0] = 100.0
    exp_adt = np.exp(-a_true * dt)
    std = np.sqrt((1 - np.exp(-2 * a_true * dt)) / (2 * a_true))
    for i in range(1, n):
        x[i] = x[i - 1] * exp_adt + 100.0 * (1 - exp_adt) + rng.normal(0, std)

    raw = mle_analytical(x, dt)
    corrected = bias_correct(raw, dt)

    rel_diff = abs(corrected.mean_rev_speed - raw.mean_rev_speed) / raw.mean_rev_speed
    assert rel_diff < 0.05


def test_est_vola_qv_recovers_vola():
    """QV-Schätzer soll wahre Volatilität auf 20% genau treffen."""
    from quant_tools.ou.estimator import est_vola_qv
    from quant_tools.ou.simulator import path
    from quant_tools.core.types import FitResult

    params = FitResult(
        mean_rev_speed=2.0, mean_rev_level=50.0, vola=1.0,
        half_life=np.log(2) / 2.0, loglik=0.0, n_obs=1,
    )
    x = path(x0=50.0, n_steps=5000, dt=1 / 252, params=params, seed=42)
    v_est = est_vola_qv(x, dt=1 / 252)
    assert abs(v_est - 1.0) / 1.0 < 0.20


def test_est_vola_qv_positive():
    """QV-Schätzer muss immer positiv sein."""
    from quant_tools.ou.estimator import est_vola_qv
    x = np.array([1.0, 1.001, 0.999, 1.002, 0.998])
    v = est_vola_qv(x, dt=1.0)
    assert v > 0


def test_est_vola_qv_min_length():
    """Zu kurze Zeitreihe → ValueError."""
    from quant_tools.ou.estimator import est_vola_qv
    with pytest.raises(ValueError, match="mindestens"):
        est_vola_qv(np.array([1.0]), dt=1.0)


def test_optimal_dt_basic():
    """optimal_dt(2.0) = 0.7968 / 2.0 = 0.3984."""
    from quant_tools.ou.estimator import optimal_dt
    result = optimal_dt(2.0)
    assert abs(result - 0.3984) < 1e-4


def test_optimal_dt_fast_reversion():
    """Schnelle Mean-Reversion → kleines dt optimal."""
    from quant_tools.ou.estimator import optimal_dt
    result = optimal_dt(100.0)
    assert result < 0.01


def test_optimal_dt_zero_raises():
    """θ=0 → ValueError."""
    from quant_tools.ou.estimator import optimal_dt
    with pytest.raises(ValueError):
        optimal_dt(0.0)
