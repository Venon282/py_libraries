import pytest
from py_libraries.type.lst import smoothMiddle


def test_smoothMiddle_odd_window_known_values() -> None:
    result = smoothMiddle([1, 2, 3, 4, 5], window=3)

    assert result == pytest.approx([1.5, 2, 3, 4, 4.5])


def test_smoothMiddle_even_window_takes_extra_element_on_the_right() -> None:
    result = smoothMiddle([1, 2, 3, 4, 5], window=4)

    assert result == pytest.approx([2, 2.5, 3.5, 4, 4.5])


@pytest.mark.parametrize('window', [2, 3, 4, 5, 6, 7])
def test_smoothMiddle_constant_signal_is_unchanged(window: int) -> None:
    signal = [4.0] * 10

    assert smoothMiddle(signal, window=window) == pytest.approx(signal)


def test_smoothMiddle_linear_signal_is_preserved_away_from_the_borders() -> None:
    signal = list(range(20))

    result = smoothMiddle(signal, window=5)

    assert result[2:-2] == pytest.approx(signal[2:-2])


def test_smoothMiddle_sequence_shorter_than_window_uses_available_values() -> None:
    assert smoothMiddle([1, 2], window=5) == pytest.approx([1.5, 1.5])


def test_smoothMiddle_empty_input_returns_empty_list() -> None:
    assert smoothMiddle([]) == []
