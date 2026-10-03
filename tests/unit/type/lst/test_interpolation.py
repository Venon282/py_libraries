import numpy as np
import pytest
from py_libraries.type.lst import interpolation

X = [0.0, 1.0, 2.0, 3.0]
Y = [10 * value for value in X]


def test_interpolation_between_nodes_follows_linear_function() -> None:
    assert interpolation([0.5, 1.5, 2.5], X, Y) == pytest.approx([5.0, 15.0, 25.0])


def test_interpolation_on_nodes_returns_original_values() -> None:
    assert interpolation(X, X, Y) == pytest.approx(Y)


def test_interpolation_out_of_range_targets_take_edge_values() -> None:
    assert interpolation([-1.0, 4.0], X, Y) == [0.0, 30.0]


def test_interpolation_one_target_per_interval_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 8))
    y = rng.uniform(-5, 5, 8)
    targets = (x[:-1] + x[1:]) / 2

    result = interpolation(targets.tolist(), x.tolist(), y.tolist())

    assert result == pytest.approx(np.interp(targets, x, y))


def test_interpolation_without_targets_returns_empty_list() -> None:
    assert interpolation([], X, Y) == []
