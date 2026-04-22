import numpy as np
import pandas as pd
import pytest
from quant_tools.ou.estimator import RollingOU


def test_rolling_output_length():
    x = np.random.default_rng(0).normal(100, 1, 500)
    r = RollingOU(window=100)
    result = r.fit(x, dt=1 / 252)
    assert len(result) == len(x)
    assert result["half_life"].isna().sum() == 390


def test_rolling_columns():
    x = np.random.default_rng(1).normal(0, 1, 300)
    r = RollingOU(window=50)
    df = r.fit(x, dt=1.0)
    assert set(df.columns) == {"mean_rev_speed", "mean_rev_level", "vola", "half_life"}


def test_rolling_accepts_pandas_series():
    s = pd.Series(np.random.default_rng(2).normal(50, 2, 400))
    r = RollingOU(window=80)
    df = r.fit(s, dt=1 / 252)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 400


def test_rolling_window_too_small_raises():
    with pytest.raises(ValueError, match="window"):
        RollingOU(window=2)
