import numpy as np
from quant_tools.core.types import FitResult


def path(
    x0: float,
    n_steps: int,
    dt: float,
    params: FitResult,
    seed: int | None = None,
) -> np.ndarray:
    """Simuliert einen OU-Pfad mit exakter Transition (nicht Euler).

    Args:
        x0:      Startwert
        n_steps: Anzahl Schritte
        dt:      Zeitschritt in Jahren (z.B. 1/252 für Tages-Bars)
        params:  FitResult mit mean_rev_speed, mean_rev_level, vola
        seed:    Zufalls-Seed für Reproduzierbarkeit
    """
    rng = np.random.default_rng(seed)
    a = params.mean_rev_speed
    mu = params.mean_rev_level
    v = params.vola

    exp_adt = np.exp(-a * dt)
    if a > 1e-10:
        trans_std = v * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    else:
        trans_std = v * np.sqrt(dt)

    x = np.empty(n_steps)
    x[0] = x0
    noise = rng.normal(0.0, trans_std, size=n_steps - 1)
    for i in range(1, n_steps):
        x[i] = x[i - 1] * exp_adt + mu * (1 - exp_adt) + noise[i - 1]
    return x


def noise_from_path(
    x: np.ndarray,
    dt: float,
    params: FitResult,
) -> np.ndarray:
    """Extrahiert standardisierte Residuen Z_t ~ N(0,1) aus OU-Pfad.

    Args:
        x:      Beobachteter Pfad
        dt:     Zeitschritt
        params: Geschätzte Parameter

    Returns:
        Array der Länge len(x)-1 mit standardisierten Residuen
    """
    a = params.mean_rev_speed
    mu = params.mean_rev_level
    v = params.vola

    exp_adt = np.exp(-a * dt)
    expected = x[:-1] * exp_adt + mu * (1 - exp_adt)

    if a > 1e-10:
        trans_std = v * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    else:
        trans_std = v * np.sqrt(dt)

    return (x[1:] - expected) / trans_std
