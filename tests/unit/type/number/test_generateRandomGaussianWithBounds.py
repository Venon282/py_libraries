"""Unit tests for generateRandomGaussianWithBounds."""

import numpy as np
import pytest

from py_libraries.type.number import generateRandomGaussianWithBounds


def patchNormal(
    monkeypatch: pytest.MonkeyPatch, draws: list[float]
) -> list[dict[str, float]]:
    """Replace `np.random.normal` by a function returning scripted values.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        draws: Values returned by the successive calls.

    Returns:
        The list that records the `loc` and `scale` of each call.
    """
    calls: list[dict[str, float]] = []
    values = iter(draws)

    def fakeNormal(loc: float, scale: float) -> float:
        calls.append({'loc': loc, 'scale': scale})
        return next(values)

    monkeypatch.setattr(np.random, 'normal', fakeNormal)
    return calls


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('retry', [True, False])
def test_result_stays_within_the_bounds(seed: int, retry: bool) -> None:
    """Return a value inside the bounds, even when the distribution is wider.

    Args:
        seed: Seed of the random generator.
        retry: Whether the draws are repeated or only clamped.
    """
    rng = np.random.default_rng(seed)

    for _ in range(200):
        value = generateRandomGaussianWithBounds(
            2.0, 5.0, std=2.0, retry=retry, rng=rng
        )

        assert 2.0 <= value <= 5.0


def test_draws_are_centered_on_the_midpoint_by_default() -> None:
    """Get a sample mean close to the midpoint of the bounds."""
    rng = np.random.default_rng(0)

    values = [generateRandomGaussianWithBounds(0, 6, rng=rng) for _ in range(5000)]

    assert np.mean(values) == pytest.approx(3.0, abs=0.1)


def test_mean_is_honored() -> None:
    """Get a sample mean close to the requested mean."""
    rng = np.random.default_rng(0)

    values = [
        generateRandomGaussianWithBounds(0, 10, mean=2.0, std=0.5, rng=rng)
        for _ in range(2000)
    ]

    assert np.mean(values) == pytest.approx(2.0, abs=0.1)


def test_same_seed_gives_the_same_value() -> None:
    """Reproduce a draw from two generators with the same seed."""
    first = generateRandomGaussianWithBounds(0, 1, rng=np.random.default_rng(7))
    second = generateRandomGaussianWithBounds(0, 1, rng=np.random.default_rng(7))

    assert first == second


@pytest.mark.parametrize(
    ('mean', 'std', 'dispersion', 'expected_loc', 'expected_scale'),
    [
        (None, None, 3, 3.0, 1.0),
        (None, None, 6, 3.0, 0.5),
        (4.0, None, 3, 4.0, 4.0 / 3),
        (None, 0.25, 3, 3.0, 0.25),
        (1.0, 2.0, 3, 1.0, 2.0),
    ],
)
def test_default_mean_and_std(
    monkeypatch: pytest.MonkeyPatch,
    mean: float | None,
    std: float | None,
    dispersion: int,
    expected_loc: float,
    expected_scale: float,
) -> None:
    """Derive the missing mean and std from the bounds and the dispersion.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mean: Mean passed to the function.
        std: Standard deviation passed to the function.
        dispersion: Dispersion passed to the function.
        expected_loc: Expected mean given to the distribution.
        expected_scale: Expected std given to the distribution.
    """
    calls = patchNormal(monkeypatch, [3.0])

    generateRandomGaussianWithBounds(0, 6, mean=mean, std=std, dispersion=dispersion)

    assert calls == [
        {'loc': pytest.approx(expected_loc), 'scale': pytest.approx(expected_scale)}
    ]


def test_retry_draws_again_until_the_value_is_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip the draws outside the bounds and return the first one inside.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    calls = patchNormal(monkeypatch, [10.0, -5.0, 2.5, 3.0])

    value = generateRandomGaussianWithBounds(0, 5, retry=True)

    assert value == 2.5
    assert len(calls) == 3


@pytest.mark.parametrize(('draw', 'expected'), [(10.0, 5.0), (-5.0, 0.0), (2.5, 2.5)])
def test_without_retry_the_first_draw_is_clamped(
    monkeypatch: pytest.MonkeyPatch, draw: float, expected: float
) -> None:
    """Clamp the only draw to the bounds.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        draw: Value returned by the distribution.
        expected: Expected clamped value.
    """
    calls = patchNormal(monkeypatch, [draw])

    assert generateRandomGaussianWithBounds(0, 5, retry=False) == expected
    assert len(calls) == 1


def test_result_is_a_builtin_float_even_when_a_bound_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a float when the integer bound is the clamped result.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    patchNormal(monkeypatch, [-100.0])

    value = generateRandomGaussianWithBounds(0, 10, retry=False)

    assert type(value) is float
    assert value == 0.0


@pytest.mark.parametrize('bound', [3, 3.5, -2])
def test_equal_bounds_return_that_value(bound: float) -> None:
    """Return the bound itself when the range is a single point.

    Args:
        bound: Value of both bounds.
    """
    assert generateRandomGaussianWithBounds(bound, bound) == bound


@pytest.mark.parametrize('use_generator', [True, False])
def test_reversed_bounds_raise_value_error(use_generator: bool) -> None:
    """Raise ValueError when the default std is negative.

    Args:
        use_generator: Whether a random generator is passed or the global state used.
    """
    rng = np.random.default_rng(0) if use_generator else None

    with pytest.raises(ValueError):
        generateRandomGaussianWithBounds(5, 1, rng=rng)


def test_global_random_state_is_used_without_generator() -> None:
    """Follow the seed of the global numpy state when no generator is given."""
    np.random.seed(3)
    first = generateRandomGaussianWithBounds(0, 1)
    np.random.seed(3)
    second = generateRandomGaussianWithBounds(0, 1)

    assert first == second
