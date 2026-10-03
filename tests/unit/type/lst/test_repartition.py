import numpy as np
import pytest
from py_libraries.type.lst import repartition


def test_repartition_shrinks_second_table_to_reach_the_proportion() -> None:
    data1 = list(range(6))
    data2 = list(range(100, 106))

    part1, part2 = repartition(data1, data2, 0.75)

    assert part1 == data1
    assert part2 == data2[:2]
    assert len(part1) / (len(part1) + len(part2)) == pytest.approx(0.75)


def test_repartition_shrinks_first_table_to_reach_the_proportion() -> None:
    data1 = list(range(10))
    data2 = list(range(100, 104))

    part1, part2 = repartition(data1, data2, 0.5)

    assert part1 == data1[:4]
    assert part2 == data2


def test_repartition_already_balanced_tables_are_unchanged() -> None:
    data1 = list(range(10))
    data2 = list(range(100, 110))

    part1, part2 = repartition(data1, data2, 0.5)

    assert part1 == data1
    assert part2 == data2


def test_repartition_numpy_inputs_are_sliced_like_lists() -> None:
    part1, part2 = repartition(np.arange(6), np.arange(100, 106), 0.75)

    np.testing.assert_array_equal(part1, np.arange(6))
    np.testing.assert_array_equal(part2, np.arange(100, 102))
