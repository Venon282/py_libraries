import numpy as np
import pytest
from py_libraries.type.lst import describe, describeValues


def test_describe_prints_one_line_per_statistic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    describe([1, 2, 3])

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(describeValues([1, 2, 3]))
    assert lines[0] == 'shape: (3,)'


def test_describe_rounds_to_n_decimals(capsys: pytest.CaptureFixture[str]) -> None:
    describe([0, 0, 1], n_decimals=2)

    assert 'mean: 0.33' in capsys.readouterr().out.splitlines()


def test_describe_without_n_decimals_does_not_round(
    capsys: pytest.CaptureFixture[str],
) -> None:
    describe([0, 0, 1], n_decimals=None)

    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith('mean: 0.3333333') for line in lines)


def test_describe_large_values_use_scientific_notation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    describe([1e6, 2e6])

    assert 'max: 2e+06' in capsys.readouterr().out.splitlines()


def test_describe_small_values_use_scientific_notation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    describe([1e-6, 2e-6])

    assert 'max: 2e-06' in capsys.readouterr().out.splitlines()


def test_describe_string_array(capsys: pytest.CaptureFixture[str]) -> None:
    describe(np.array(['a', 'b', 'a']))

    lines = capsys.readouterr().out.splitlines()
    assert "most_common: [('a', 2), ('b', 1)]" in lines
