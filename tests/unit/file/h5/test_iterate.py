"""Unit tests for iterate."""

import h5py
from py_libraries.file.h5 import iterate


def _paths(h5: h5py.File | h5py.Group, path: str | list[str]) -> set[str]:
    """Collect the internal paths yielded by `iterate`.

    Args:
        h5: The group or file to traverse.
        path: The path pattern to traverse.

    Returns:
        The set of yielded paths.
    """
    return {internal_path for internal_path, _ in iterate(h5, path)}


def test_exact_path_returns_the_object(h5file: h5py.File) -> None:
    """Return the object reached by an exact path."""
    result = dict(iterate(h5file, 'raw/a'))

    assert set(result) == {'raw/a'}
    assert isinstance(result['raw/a'], h5py.Dataset)


def test_single_group_name_returns_the_group(h5file: h5py.File) -> None:
    """Return the group itself when the path is its name."""
    result = dict(iterate(h5file, 'raw'))

    assert set(result) == {'raw'}
    assert isinstance(result['raw'], h5py.Group)


def test_star_returns_the_direct_children(h5file: h5py.File) -> None:
    """Return every direct child of a group with `*`."""
    assert _paths(h5file, 'raw/*') == {'raw/a', 'raw/deep', 'raw/s'}


def test_double_star_returns_the_whole_subtree(h5file: h5py.File) -> None:
    """Return every group and dataset of the tree with `**`."""
    expected = {'empty', 'raw', 'raw/a', 'raw/s', 'raw/deep', 'raw/deep/b', 'top'}

    assert _paths(h5file, '**') == expected


def test_double_star_from_a_subgroup_stays_in_the_subtree(h5file: h5py.File) -> None:
    """Return the subtree of a subgroup with `**`."""
    assert _paths(h5file, 'raw/**') == {'raw/a', 'raw/s', 'raw/deep', 'raw/deep/b'}


def test_double_star_in_the_middle_reaches_a_nested_dataset(h5file: h5py.File) -> None:
    """Reach a dataset nested under a subgroup with `**` in the middle."""
    assert _paths(h5file, 'raw/**/b') == {'raw/deep/b'}


def test_bracket_subset_returns_the_listed_elements(h5file: h5py.File) -> None:
    """Return only the elements listed between the brackets."""
    assert _paths(h5file, '[top, raw]') == {'top', 'raw'}


def test_tilde_bracket_excludes_the_listed_elements(h5file: h5py.File) -> None:
    """Return every element except the ones listed."""
    assert _paths(h5file, '~[top]') == {'empty', 'raw'}


def test_a_list_of_paths_is_traversed_in_turn(h5file: h5py.File) -> None:
    """Traverse every path of a list and combine the results."""
    assert _paths(h5file, ['top', 'raw/a']) == {'top', 'raw/a'}


def test_a_path_that_matches_nothing_yields_nothing(h5file: h5py.File) -> None:
    """Yield nothing when the path does not exist."""
    assert _paths(h5file, 'nonexistent') == set()


def test_a_mixed_wildcard_and_exact_path(h5file: h5py.File) -> None:
    """Combine a wildcard and an exact name in one path."""
    assert _paths(h5file, 'raw/*') == {'raw/a', 'raw/deep', 'raw/s'}
    assert _paths(h5file, 'raw/deep/b') == {'raw/deep/b'}
