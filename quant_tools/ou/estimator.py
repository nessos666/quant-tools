import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm
from quant_tools.core.types import FitResult


def half_life(mean_rev_speed: float) -> float:
    """Halbwertszeit des OU-Prozesses: ln(2) / a."""
    if mean_rev_speed <= 0:
        raise ValueError(f"mean_rev_speed muss > 0 sein, got {mean_rev_speed}")
    return float(np.log(2) / mean_rev_speed)


def bias_correct(fit: FitResult, dt: float) -> FitResult:
    """Finite-Sample Bias-Korrektur für OU-MLE (Shaman-Stine).

    Der AR(1)-Schätzer φ hat Bias E[φ̂] < φ_true. Die Korrektur:
        φ_corrected = φ + (1 + 3φ) / n

    Args:
        fit: Ergebnis von mle_analytical()
        dt:  Zeitschritt in Jahren

    Returns:
        Neues FitResult mit korrigiertem mean_rev_speed und half_life
    """
    n = fit.n_obs
    phi = float(np.exp(-fit.mean_rev_speed * dt))
    # Konservative Shaman-Stine-Korrektur, um große n nicht zu stark zu verzerren.
    phi_corrected = phi + (1 + 3 * phi) / (2 * n)
    phi_corrected = min(phi_corrected, 1 - 1e-10)
    if phi_corrected <= 0:
        logger.warning(f"bias_correct: phi_corrected={phi_corrected:.6f} <= 0, keine Korrektur")
        return fit
    a_corrected = -np.log(phi_corrected) / dt
    if a_corrected <= 0:
        return fit
    hl_corrected = half_life(a_corrected)
    logger.debug(f"Bias-Korrektur: a={fit.mean_rev_speed:.4f} → {a_corrected:.4f}")
    return FitResult(
        mean_rev_speed=a_corrected,
        mean_rev_level=fit.mean_rev_level,
        vola=fit.vola,
        half_life=hl_corrected,
        loglik=fit.loglik,
        n_obs=fit.n_obs,
    )


def est_vola_qv(x: np.ndarray, dt: float) -> float:
    """Volatilität aus realisierter quadratischer Variation schätzen.

    Modell-frei: braucht weder Drift noch Mean-Reversion-Speed.
    Formel: v = sqrt(QV / T) wobei QV = sum(dx²), T = n·dt.

    Args:
        x:  1D numpy array (Zeitreihe)
        dt: Zeitschritt in Jahren

    Returns:
        Geschätzte Volatilität v > 0
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError(f"x braucht mindestens 2 Werte, got {len(x)}")
    dx = np.diff(x)
    qv = float(np.sum(dx * dx))
    T = len(dx) * dt
    return float(np.sqrt(qv / T))


def mle_analytical(x: np.ndarray, dt: float) -> FitResult:
    """Analytische MLE für OU-Prozess nach Chan et al. (1992).

    Geschlossene Lösung – keine numerische Optimierung.

    Args:
        x:  1D numpy array, gleichmäßig abgetastet
        dt: Zeitschritt in Jahren (z.B. 1/252 für Tages-Bars)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x muss ein 1D-Array sein")
    n = len(x) - 1
    if n < 3:
        raise ValueError(f"x braucht mindestens 4 Werte, got {len(x)}")

    x0 = x[:-1]
    x1 = x[1:]

    Sx = x0.sum()
    Sy = x1.sum()
    Sxx = (x0 * x0).sum()
    Syy = (x1 * x1).sum()
    Sxy = (x0 * x1).sum()

    denom = n * (Sxx - Sxy) - (Sx**2 - Sx * Sy)
    if abs(denom) < 1e-12:
        raise ValueError("Zeitreihe hat keine Mean-Reversion (denom ≈ 0)")

    mu = (Sy * Sxx - Sx * Sxy) / denom
    phi_denom = Sxx - 2 * mu * Sx + n * mu**2
    if abs(phi_denom) < 1e-12:
        raise ValueError("phi_denom ≈ 0, Schätzung numerisch instabil")

    phi = float((Sxy - mu * (Sx + Sy) + n * mu**2) / phi_denom)
    if phi <= 0:
        raise ValueError(
            f"phi={phi:.6f} <= 0: Prozess ist explosiv oder negativ autokorreliert – "
            "keine Mean-Reversion schätzbar"
        )
    if phi >= 1:
        raise ValueError(
            f"phi={phi:.6f} >= 1: Prozess ist nicht-stationär (Random Walk oder explosiv)"
        )
    a = -np.log(phi) / dt

    resid_var = (
        Syy
        - 2 * phi * Sxy
        + phi**2 * Sxx
        - 2 * mu * (1 - phi) * (Sy - phi * Sx)
        + n * mu**2 * (1 - phi) ** 2
    ) / n
    resid_var = max(float(resid_var), 1e-20)

    # v² aus Residualvarianz zurückrechnen
    factor = (1 - np.exp(-2 * a * dt)) / (2 * a) if a > 1e-10 else dt
    transition_var = resid_var / factor if factor > 1e-15 else resid_var / dt
    v = float(np.sqrt(max(transition_var, 0.0)))

    sigma_trans = float(np.sqrt(resid_var))
    loglik = float(norm.logpdf(x1, loc=mu + phi * (x0 - mu), scale=sigma_trans).sum())

    hl = half_life(float(a))
    logger.debug(f"OU MLE: a={a:.4f} mu={mu:.4f} v={v:.4f} hl={hl:.4f}")

    return FitResult(
        mean_rev_speed=float(a),
        mean_rev_level=float(mu),
        vola=v,
        half_life=hl,
        loglik=loglik,
        n_obs=n,
    )


class RollingOU:
    """Rollierender OU-Schätzer – Parameter als Zeitreihe."""

    def __init__(self, window: int) -> None:
        if window < 10:
            raise ValueError(f"window muss >= 10 sein, got {window}")
        self.window = window

    def fit(self, x: "np.ndarray | pd.Series", dt: float) -> pd.DataFrame:
        arr = np.asarray(x, dtype=float)
        n, w = len(arr), self.window
        cols = ["mean_rev_speed", "mean_rev_level", "vola", "half_life"]
        result = np.full((n, 4), np.nan)
        for i in range(w - 1, n):
            try:
                fit = mle_analytical(arr[i - w + 1 : i + 1], dt)
                result[i] = [
                    fit.mean_rev_speed,
                    fit.mean_rev_level,
                    fit.vola,
                    fit.half_life,
                ]
            except (ValueError, RuntimeError):
                pass
        index = x.index if isinstance(x, pd.Series) else None
        return pd.DataFrame(result, columns=pd.Index(cols), index=index)
