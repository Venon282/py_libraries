import pytest
from py_libraries.type.string import toPlurialFr


@pytest.mark.parametrize(
    ('word', 'expected'),
    [
        ('chat', 'chats'),
        ('cheval', 'chevaux'),
        ('bateau', 'bateaux'),
        ('jeu', 'jeux'),
        ('travail', 'travaux'),
        ('rail', 'rails'),
    ],
)
def test_toPlurialFr_regular_rules(word: str, expected: str) -> None:
    assert toPlurialFr(word) == expected


@pytest.mark.parametrize('word', ['souris', 'prix', 'nez'])
def test_toPlurialFr_words_ending_with_s_x_or_z_are_unchanged(word: str) -> None:
    assert toPlurialFr(word) == word


def test_toPlurialFr_empty_word_raises_index_error() -> None:
    with pytest.raises(IndexError):
        toPlurialFr('')
