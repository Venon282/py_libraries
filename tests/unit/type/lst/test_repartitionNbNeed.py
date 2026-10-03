import pytest
from py_libraries.type.lst import repartition, repartitionNbNeed


@pytest.mark.parametrize(
    ('size1', 'size2', 'proportion', 'expected'),
    [
        (6, 6, 0.75, (6, 2)),
        (10, 4, 0.5, (4, 4)),
        (10, 10, 0.5, (10, 10)),
    ],
)
def test_repartitionNbNeed_known_values(
    size1: int, size2: int, proportion: float, expected: tuple[int, int]
) -> None:
    data1 = [0] * size1
    data2 = [0] * size2

    assert repartitionNbNeed(data1, data2, proportion) == expected


@pytest.mark.parametrize(
    ('size1', 'size2', 'proportion'),
    [(6, 6, 0.75), (10, 4, 0.5), (10, 10, 0.5), (3, 20, 0.25), (20, 3, 0.6)],
)
def test_repartitionNbNeed_matches_the_sizes_returned_by_repartition(
    size1: int, size2: int, proportion: float
) -> None:
    data1 = [0] * size1
    data2 = [0] * size2

    part1, part2 = repartition(data1, data2, proportion)

    assert repartitionNbNeed(data1, data2, proportion) == (len(part1), len(part2))
