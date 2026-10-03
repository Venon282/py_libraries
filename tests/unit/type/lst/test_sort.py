from py_libraries.type.lst import sort


def test_sort_orders_embedded_numbers_numerically() -> None:
    names = ['file10', 'file2', 'file1']

    sort(names)

    assert names == ['file1', 'file2', 'file10']


def test_sort_text_is_case_insensitive() -> None:
    names = ['b', 'A', 'c']

    sort(names)

    assert names == ['A', 'b', 'c']


def test_sort_numeric_blocks_come_before_text_blocks() -> None:
    names = ['abc', '10', '9']

    sort(names)

    assert names == ['9', '10', 'abc']


def test_sort_matches_plain_sort_for_zero_padded_numbers() -> None:
    names = ['b02', 'a10', 'a02']

    sort(names)

    assert names == sorted(names)
    assert names == ['a02', 'a10', 'b02']


def test_sort_empty_list_stays_empty() -> None:
    names: list[str] = []

    sort(names)

    assert names == []
