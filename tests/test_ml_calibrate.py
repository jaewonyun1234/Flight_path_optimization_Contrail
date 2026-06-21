"""Calibration: isotonic lowers ECE; conformal interval covers ~(1-alpha)."""

import numpy as np

from contrail_ml.calibrate import (
    ConformalRHi,
    ProbabilityCalibrator,
    expected_calibration_error,
)


def test_isotonic_lowers_ece_on_heldout():
    rng = np.random.default_rng(0)
    n = 4000
    # Latent score -> true label; observed prob is a miscalibrated (squashed)
    # version of the latent, so calibration has real work to do.
    latent = rng.normal(size=n)
    y = (latent + rng.normal(scale=0.5, size=n) > 0).astype(int)
    p_true = 1.0 / (1.0 + np.exp(-latent))
    p_miscal = p_true ** 2  # systematically under-confident

    half = n // 2
    cal = ProbabilityCalibrator(method="isotonic").fit(p_miscal[:half], y[:half])
    p_cal = cal.transform(p_miscal[half:])

    ece_before = expected_calibration_error(p_miscal[half:], y[half:])
    ece_after = expected_calibration_error(p_cal, y[half:])
    assert ece_after < ece_before
    assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))


def test_conformal_interval_covers_target():
    rng = np.random.default_rng(1)
    n = 5000
    rhi_hat = rng.uniform(40, 140, size=n)
    rhi_true = rhi_hat + rng.normal(scale=6.0, size=n)

    half = n // 2
    conf = ConformalRHi(alpha=0.1).fit(rhi_true[:half], rhi_hat[:half])
    lo, hi = conf.interval(rhi_hat[half:])
    covered = np.mean((rhi_true[half:] >= lo) & (rhi_true[half:] <= hi))
    assert conf.half_width > 0
    assert covered >= 0.85  # ~90% target, allow sampling slack
