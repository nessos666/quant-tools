from pydantic import BaseModel, field_validator


class FitResult(BaseModel):
    """Ergebnis eines Schätzers (OU, GARCH, etc.)"""

    mean_rev_speed: float
    mean_rev_level: float
    vola: float
    half_life: float
    loglik: float
    n_obs: int

    @field_validator("vola")
    @classmethod
    def vola_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"vola muss >= 0 sein, got {v}")
        return v

    @field_validator("mean_rev_speed")
    @classmethod
    def speed_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"mean_rev_speed muss >= 0 sein, got {v}")
        return v
