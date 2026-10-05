import random

import pytest
from py_libraries.type.string import is1SwapAreEqual


@pytest.mark.parametrize(
    ('s1', 's2', 'expected'),
    [
        ('bank', 'kanb', True),
        ('attack', 'defend', False),
        ('kelb', 'kelb', True),
        ('abcd', 'dcba', False),
        ('ab', 'abc', False),
        ('', '', True),
        ('ab', 'ba', True),
        ('abc', 'abd', False),
    ],
)
def test_is1SwapAreEqual_known_values(s1: str, s2: str, expected: bool) -> None:
    assert is1SwapAreEqual(s1, s2) is expected


def test_is1SwapAreEqual_a_real_swap_is_always_detected() -> None:
    rng = random.Random(0)

    for _ in range(100):
        chars = [rng.choice('abc') for _ in range(rng.randint(2, 8))]
        i, j = rng.sample(range(len(chars)), 2)
        swapped = chars.copy()
        swapped[i], swapped[j] = swapped[j], swapped[i]

        assert is1SwapAreEqual(''.join(chars), ''.join(swapped))


def test_is1SwapAreEqual_is_symmetric() -> None:
    rng = random.Random(1)

    for _ in range(100):
        s1 = ''.join(rng.choice('ab') for _ in range(4))
        s2 = ''.join(rng.choice('ab') for _ in range(4))

        assert is1SwapAreEqual(s1, s2) == is1SwapAreEqual(s2, s1)
