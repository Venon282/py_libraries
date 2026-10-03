from collections import Counter
from typing import cast

import numpy as np
import pytest
from py_libraries.type.lst import describeValues


def test_describeValues_statistics_of_symmetric_values() -> None:
    stats = describeValues([1, 2, 3, 4, 5])

    assert stats['shape'] == (5,)
    assert stats['total_count'] == 5
    assert stats['count'] == 5
    assert stats['min'] == 1
    assert stats['max'] == 5
    assert stats['range'] == 4
    assert stats['mean'] == pytest.approx(3.0)
    assert stats['median'] == pytest.approx(3.0)
    assert stats['25%'] == pytest.approx(2.0)
    assert stats['75%'] == pytest.approx(4.0)
    assert stats['IQR'] == pytest.approx(2.0)
    assert stats['var'] == pytest.approx(2.0)
    assert stats['std'] == pytest.approx(np.sqrt(2.0))
    assert stats['unique_count'] == 5
    assert stats['skewness'] == pytest.approx(0.0, abs=1e-12)


def test_describeValues_counts_non_finite_values_separately() -> None:
    stats = describeValues([1.0, 2.0, 0.0, np.nan, np.inf, -np.inf])

    assert stats['total_count'] == 6
    assert stats['count'] == 3
    assert stats['nan_count'] == 1
    assert stats['pos_inf_count'] == 1
    assert stats['neg_inf_count'] == 1
    assert stats['zero_count'] == 1
    assert stats['nan_rate'] == pytest.approx(1 / 6)
    assert stats['zero_count_rate'] == pytest.approx(1 / 3)
    assert stats['mean'] == pytest.approx(1.0)
    assert stats['min'] == 0.0
    assert stats['max'] == 2.0


def test_describeValues_chunked_moments_match_numpy() -> None:
    data = np.random.default_rng(0).normal(5.0, 2.0, size=1000)

    stats = describeValues(data, chunk_size=7)

    assert stats['mean'] == pytest.approx(data.mean())
    assert stats['var'] == pytest.approx(data.var())
    assert stats['std'] == pytest.approx(data.std())
    assert stats['min'] == np.min(data)
    assert stats['max'] == np.max(data)


def test_describeValues_chunk_size_does_not_change_the_result() -> None:
    data = np.random.default_rng(1).normal(size=500)

    small = describeValues(data, chunk_size=13)
    large = describeValues(data, chunk_size=10_000)

    assert small.keys() == large.keys()
    for key, value in large.items():
        assert small[key] == pytest.approx(value)


def test_describeValues_sampling_keeps_exact_moments() -> None:
    data = np.arange(100, dtype=float)

    stats = describeValues(data, chunk_size=10, sample_limit=50)

    assert stats['count'] == 100
    assert stats['mean'] == pytest.approx(49.5)
    assert stats['std'] == pytest.approx(data.std())
    assert stats['min'] == 0.0
    assert stats['max'] == 99.0
    assert stats['unique_count'] == 50


def test_describeValues_all_nan_gives_nan_statistics() -> None:
    stats = describeValues(np.array([np.nan, np.nan]))

    assert stats['count'] == 0
    assert stats['nan_count'] == 2
    assert np.isnan(stats['mean'])
    assert np.isnan(stats['median'])


def test_describeValues_empty_array() -> None:
    stats = describeValues(np.array([]))

    assert stats['total_count'] == 0
    assert stats['count'] == 0
    assert np.isnan(stats['nan_rate'])


def test_describeValues_multidimensional_array_is_flattened() -> None:
    stats = describeValues(np.arange(6).reshape(2, 3))

    assert stats['shape'] == (2, 3)
    assert stats['count'] == 6
    assert stats['mean'] == pytest.approx(2.5)


def test_describeValues_string_array_statistics() -> None:
    stats = describeValues(np.array(['a', 'bb', 'a']))

    assert stats == {
        'shape': (3,),
        'total_count': 3,
        'unique_count': 2,
        'min_length': 1,
        'max_length': 2,
        'most_common': [('a', 2), ('bb', 1)],
    }


def test_describeValues_string_array_chunk_size_does_not_change_the_result() -> None:
    values = np.array(['a', 'bb', 'a', 'ccc', 'bb', 'a'])

    assert describeValues(values, chunk_size=1) == describeValues(values)


def test_describeValues_string_array_most_common_limit() -> None:
    values = np.array(['a', 'a', 'b', 'c'])

    stats = describeValues(values, str_most_common_limit=1)

    assert stats['most_common'] == [('a', 2)]


def test_describeValues_string_array_without_limit_returns_the_counter() -> None:
    values = np.array(['a', 'a', 'b', 'c'])

    stats = describeValues(values, str_most_common_limit=None)

    assert stats['most_common'] == Counter({'a': 2, 'b': 1, 'c': 1})


def test_describeValues_invalid_most_common_limit_raises_type_error() -> None:
    with pytest.raises(TypeError):
        describeValues(np.array(['a']), str_most_common_limit=cast(int, '2'))
