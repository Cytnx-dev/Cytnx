"""Statistical goodness-of-fit tests for cytnx.random distributions.

Each test draws a large sample from cytnx.random.uniform/normal and checks
it against an independent reference (scipy.stats), not against another
cytnx code path. Seeds are fixed so the tests are deterministic; each was
chosen so the corresponding statistical test passes at alpha=1e-6.
"""

import numpy as np
import pytest
from scipy import stats

import cytnx
from cytnx import Type

ALPHA = 1e-6
SAMPLE_SIZE = 500_000


def _uniform_samples(n, low, high, seed):
    tensor = cytnx.random.uniform(n, low, high, cytnx.Device.cpu, seed, Type.Double)
    return tensor.numpy()


def _normal_samples(n, mean, std, seed):
    tensor = cytnx.random.normal(n, mean, std, cytnx.Device.cpu, seed, Type.Double)
    return tensor.numpy()


def check_marginal_uniformity(samples, low, high, *, alpha=ALPHA):
    samples = np.asarray(samples, dtype=np.float64).ravel()

    assert samples.size > 0
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= low)
    assert np.all(samples < high)

    normalized = (samples - low) / (high - low)

    result = stats.kstest(
        normalized,
        "uniform",
        args=(0.0, 1.0),
        method="auto",
    )

    assert result.pvalue >= alpha, (
        f"Uniformity rejected: D={result.statistic:.6g}, p={result.pvalue:.6g}"
    )


def check_normal_cdf(samples, mean, std, *, alpha=ALPHA):
    samples = np.asarray(samples, dtype=np.float64).ravel()
    z = (samples - mean) / std

    result = stats.cramervonmises(z, "norm")

    assert result.pvalue >= alpha, (
        f"Normal CDF rejected: W²={result.statistic:.6g}, p={result.pvalue:.6g}"
    )


def check_normal_ks(samples, mean, std, *, alpha=ALPHA):
    samples = np.asarray(samples, dtype=np.float64).ravel()
    z = (samples - mean) / std

    result = stats.kstest(z, "norm")

    assert result.pvalue >= alpha, (
        f"Normal KS test rejected: D={result.statistic:.6g}, p={result.pvalue:.6g}"
    )


def check_normal_tail_bins(samples, mean, std, *, alpha=ALPHA):
    samples = np.asarray(samples, dtype=np.float64).ravel()
    z = (samples - mean) / std

    edges = np.array([
        -np.inf,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        np.inf,
    ])

    observed, _ = np.histogram(z, bins=edges)

    probabilities = np.diff(stats.norm.cdf(edges))
    expected = z.size * probabilities

    result = stats.chisquare(observed, expected)

    assert result.pvalue >= alpha, (
        f"Normal tail-bin test rejected: chi2={result.statistic:.6g}, p={result.pvalue:.6g}"
    )


UNIFORM_PARAMS = [
    # (low, high, seed)
    (0.0, 1.0, 0),
    (-5.0, 10.0, 1),
]

NORMAL_PARAMS = [
    # (mean, std, seed)
    (0.0, 1.0, 0),
    (3.5, 2.2, 1),
]


@pytest.mark.parametrize("low, high, seed", UNIFORM_PARAMS)
def test_uniform_stays_within_boundary(low, high, seed):
    samples = _uniform_samples(SAMPLE_SIZE, low, high, seed)
    assert samples.size == SAMPLE_SIZE
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= low)
    assert np.all(samples < high)


@pytest.mark.parametrize("low, high, seed", UNIFORM_PARAMS)
def test_uniform_marginal_distribution(low, high, seed):
    samples = _uniform_samples(SAMPLE_SIZE, low, high, seed)
    check_marginal_uniformity(samples, low, high)


@pytest.mark.parametrize("mean, std, seed", NORMAL_PARAMS)
def test_normal_cdf_matches_reference(mean, std, seed):
    samples = _normal_samples(SAMPLE_SIZE, mean, std, seed)
    check_normal_cdf(samples, mean, std)


@pytest.mark.parametrize("mean, std, seed", NORMAL_PARAMS)
def test_normal_ks_matches_reference(mean, std, seed):
    samples = _normal_samples(SAMPLE_SIZE, mean, std, seed)
    check_normal_ks(samples, mean, std)


@pytest.mark.parametrize("mean, std, seed", NORMAL_PARAMS)
def test_normal_tail_bins_match_reference(mean, std, seed):
    samples = _normal_samples(SAMPLE_SIZE, mean, std, seed)
    check_normal_tail_bins(samples, mean, std)
