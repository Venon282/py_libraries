import pytest
from py_libraries.type.string import maxOccurenceAdjacentEmoji

SMILE = '\U0001f600'
GRIN = '\U0001f601'
LAUGH = '\U0001f602'
FAMILY = '\U0001f468\u200d\U0001f469\u200d\U0001f467'


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        (f'a{SMILE}{GRIN}b{LAUGH}', 2),
        (SMILE + GRIN + LAUGH, 3),
        (f'{SMILE}a{GRIN}a{LAUGH}', 1),
        (SMILE, 1),
        ('abc', 0),
        ('', 0),
    ],
)
def test_maxOccurenceAdjacentEmoji_known_values(s: str, expected: int) -> None:
    assert maxOccurenceAdjacentEmoji(s) == expected


def test_maxOccurenceAdjacentEmoji_joined_sequence_counts_as_one() -> None:
    assert maxOccurenceAdjacentEmoji(FAMILY) == 1
    assert maxOccurenceAdjacentEmoji(SMILE + FAMILY + GRIN) == 3


def test_maxOccurenceAdjacentEmoji_a_non_emoji_resets_the_count() -> None:
    assert maxOccurenceAdjacentEmoji(f'{SMILE}{GRIN}x{LAUGH}') == 2
