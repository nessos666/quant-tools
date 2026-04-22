import pytest
from pydantic import ValidationError

from quant_tools.core.types import FitResult


def test_fit_result_creation():
    r = FitResult(
        mean_rev_speed=0.5,
        mean_rev_level=100.0,
        vola=0.02,
        half_life=1.386,
        loglik=-250.3,
        n_obs=1000,
    )
    assert r.mean_rev_speed == 0.5
    assert r.mean_rev_level == 100.0
    assert r.vola == 0.02
    assert abs(r.half_life - 1.386) < 1e-6
    assert r.n_obs == 1000


def test_fit_result_validation_negative_vola():
    with pytest.raises(ValidationError):
        FitResult(
            mean_rev_speed=0.5,
            mean_rev_level=100.0,
            vola=-0.01,
            half_life=1.386,
            loglik=-250.3,
            n_obs=1000,
        )


def test_fit_result_validation_zero_vola():
    with pytest.raises(ValidationError):
        FitResult(
            mean_rev_speed=0.5,
            mean_rev_level=100.0,
            vola=0.0,
            half_life=1.386,
            loglik=-250.3,
            n_obs=1000,
        )


def test_fit_result_validation_negative_speed():
    with pytest.raises(ValidationError):
        FitResult(
            mean_rev_speed=-0.1,
            mean_rev_level=100.0,
            vola=0.02,
            half_life=1.386,
            loglik=-250.3,
            n_obs=1000,
        )


def test_fit_result_validation_zero_speed():
    with pytest.raises(ValidationError):
        FitResult(
            mean_rev_speed=0.0,
            mean_rev_level=100.0,
            vola=0.02,
            half_life=1.386,
            loglik=-250.3,
            n_obs=1000,
        )


def test_fit_result_validation_negative_half_life():
    with pytest.raises(ValidationError):
        FitResult(
            mean_rev_speed=0.5,
            mean_rev_level=100.0,
            vola=0.02,
            half_life=-1.0,
            loglik=-250.3,
            n_obs=1000,
        )


def test_fit_result_validation_zero_n_obs():
    with pytest.raises(ValidationError):
        FitResult(
            mean_rev_speed=0.5,
            mean_rev_level=100.0,
            vola=0.02,
            half_life=1.386,
            loglik=-250.3,
            n_obs=0,
        )
