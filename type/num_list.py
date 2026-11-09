"""Algorithms on lists of numbers: sums, products, intervals and subarrays."""

from collections import defaultdict
from collections.abc import Sequence


def tupleSameProduct(nums: Sequence[int]) -> int:
    """Count the ordered tuples (a, b, c, d) of elements such that a * b == c * d.

    The four elements of a tuple come from four different positions of `nums`.

    Args:
        nums: Distinct positive integers. With duplicates or zeros, two pairs with
            the same product can share an element, and the result is wrong.

    Returns:
        The number of ordered tuples (a, b, c, d) with a * b == c * d.

    Example:
        >>> tupleSameProduct([2, 3, 4, 6])
        8
    """
    products: defaultdict[int, int] = defaultdict(int)
    ans = 0

    for i, num in enumerate(nums[:-1]):
        for n in nums[i + 1 :]:
            product = num * n
            # Each pair of pairs with the same product gives 8 ordered tuples
            ans += 8 * products[product]
            products[product] += 1
    return ans


def idxSumOfTwo(nums: Sequence[int], target: int) -> list[int]:
    """Find two elements whose sum is the target.

    Args:
        nums: Integers to scan.
        target: Sum to reach.

    Returns:
        The indices `[i, j]`, with `i < j`, of two elements such that
        `nums[i] + nums[j] == target`, or an empty list if there is none. The pair
        with the lowest `j` is returned.
    """
    complements: dict[int, int] = {}
    for i, num in enumerate(nums):
        if num in complements:
            return [complements[num], i]
        complements[target - num] = i
    return []


def mergeIntervals(intervals: Sequence[Sequence[int]]) -> list[list[int]]:
    """Merge the overlapping intervals.

    Intervals that only touch (`[1, 2]` and `[2, 3]`) are merged.

    Args:
        intervals: Closed intervals `[start, end]`, in any order. It is not
            modified.

    Returns:
        New intervals, sorted by start, that cover the same points as `intervals`
        without overlapping each other.

    Example:
        >>> mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]])
        [[1, 6], [8, 10], [15, 18]]
    """
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda interval: interval[0])
    merged = [list(ordered[0])]

    for start, end in ordered[1:]:
        if merged[-1][1] >= start:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def oddNumberInEvenList(nums: Sequence[int]) -> int:
    """Find the element that appears an odd number of times with a XOR.

    This is meant for a list where every element appears twice except one.

    Args:
        nums: Integers to scan.

    Returns:
        The XOR of all the elements, which is the element appearing an odd number
        of times when there is exactly one. It is 0 when every element appears an
        even number of times.
    """
    res = 0
    for num in nums:
        res ^= num
    return res


def pascalTriangle(n_rows: int) -> list[list[int]]:
    """Build the first rows of Pascal's triangle.

    Args:
        n_rows: Number of rows to build. At least the first row is always
            built, so a value below 1 gives the same result as 1.

    Returns:
        The rows of the triangle, where each inner value is the sum of the two
        values above it.

    Example:
        >>> pascalTriangle(4)
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]
    """
    triangle = [[1]]
    for _ in range(n_rows - 1):
        previous = triangle[-1]
        # The shifted copy is one element shorter: it pairs the neighbours
        sums = [a + b for a, b in zip(previous, previous[1:], strict=False)]
        triangle.append([1, *sums, 1])
    return triangle


def minQueriesForZeroArray(
    nums: Sequence[int], queries: Sequence[Sequence[int]]
) -> int:
    """Find the minimum number of queries needed to turn an array into a zero array.

    Each query `[left, right, val]` lets every element of `nums[left:right + 1]`
    be decreased by any amount up to `val`, independently of the other elements.
    The queries are used in order, from the first one.

    Args:
        nums: Non-negative integers of the initial array.
        queries: Queries `[left, right, val]`, with `0 <= left <= right <
            len(nums)`.

    Returns:
        The smallest `k` such that the first `k` queries can turn `nums` into a zero
        array, or -1 if all the queries are not enough.

    Raises:
        IndexError: If a query that has to be used has a `right` bound outside of
            `nums`.

    Example:
        >>> minQueriesForZeroArray([2, 0, 2], [[0, 2, 1], [0, 2, 1], [1, 1, 3]])
        2
    """
    # Difference array, to apply a range update in constant time. Its size is
    # len(nums) + 1 to handle the end offset of a range reaching the last element.
    diffs = [0] * (len(nums) + 1)

    query_idx = 0  # Number of queries used so far
    total_sum = 0  # Total decrease available at the current index

    for i in range(len(nums)):
        # Use more queries until the decrease available at i reaches nums[i]
        while total_sum + diffs[i] < nums[i]:
            query_idx += 1

            if query_idx > len(queries):
                return -1

            left, right, val = queries[query_idx - 1]

            # A query ending before i cannot help anymore
            if right >= i:
                # The part of the query before i is already behind us
                diffs[max(left, i)] += val
                diffs[right + 1] -= val

        total_sum += diffs[i]

    return query_idx


def maxItemsPerPerson(piles: Sequence[int], k: int) -> int:
    """Find the maximum number of items that each person can receive.

    Each pile can be divided into sub-piles, but the items of different piles
    cannot be merged. Each person receives items from a single pile, and all the
    persons receive the same number of items.

    Args:
        piles: Number of items of each pile, non-negative.
        k: Number of persons to allocate items to. It must be positive.

    Returns:
        The maximum number of items each person can receive, or 0 if the piles do
        not hold enough items to give at least one to each person.

    Example:
        >>> maxItemsPerPerson([5, 8, 6], 3)
        5
    """
    if sum(piles) < k:
        return 0

    # The answer is between 1 item and the content of the largest pile
    left, right = 1, max(piles)

    # Binary search: the more items per person, the fewer persons can be served
    while left <= right:
        middle = left + (right - left) // 2

        # Each pile serves floor(pile / middle) persons
        if sum(pile // middle for pile in piles) >= k:
            left = middle + 1
        else:
            right = middle - 1

    return right


def uniqueTriplesSumAt0(nums: Sequence[int]) -> list[list[int]]:
    """Find all unique triplets of elements that sum up to 0.

    It uses the two-pointer technique on the sorted list, so it runs in O(n^2).

    Args:
        nums: Integers to scan. It is not modified.

    Returns:
        The triplets `[a, b, c]`, each one in ascending order, without duplicates,
        sorted in lexicographic order.

    Example:
        >>> uniqueTriplesSumAt0([-1, 0, 1, 2, -1, -4])
        [[-1, -1, 2], [-1, 0, 1]]
    """
    ordered = sorted(nums)
    triples = []

    for i in range(len(ordered) - 2):
        if i > 0 and ordered[i] == ordered[i - 1]:
            continue

        left, right = i + 1, len(ordered) - 1
        while left < right:
            total = ordered[i] + ordered[left] + ordered[right]

            if total == 0:
                triples.append([ordered[i], ordered[left], ordered[right]])

                while left < right and ordered[left] == ordered[left + 1]:
                    left += 1
                while left < right and ordered[right] == ordered[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return triples


def maximumUniqueSubarray(nums: Sequence[int]) -> int:
    """Find the maximum sum of a contiguous subarray of unique elements.

    Args:
        nums: Positive integers. With negative values the sliding window can miss
            the best subarray.

    Returns:
        The maximum sum of the elements of a contiguous subarray whose elements are
        all different, 0 for an empty list.

    Example:
        >>> maximumUniqueSubarray([4, 2, 4, 5, 6])
        17

        The best subarray is `[2, 4, 5, 6]`.
    """
    res = 0
    cur_sum = 0
    left = 0
    seen: set[int] = set()

    for right in range(len(nums)):
        while nums[right] in seen:
            seen.remove(nums[left])
            cur_sum -= nums[left]
            left += 1

        cur_sum += nums[right]
        seen.add(nums[right])
        res = max(res, cur_sum)
    return res


def maxArea(height: Sequence[int]) -> int:
    """Find the container that holds the most water.

    There are `len(height)` vertical lines, the i-th one going from `(i, 0)` to
    `(i, height[i])`. A container is made of two lines and the x-axis, and it cannot
    be slanted.

    Args:
        height: Non-negative heights of the vertical lines.

    Returns:
        The maximum amount of water a container can store, 0 if there are fewer
        than two lines.

    Example:
        >>> maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7])
        49
    """
    max_area = 0
    left, right = 0, len(height) - 1

    while left < right:
        weak_link = min(height[right], height[left])
        area = (right - left) * weak_link
        max_area = max(max_area, area)

        # Only moving the lower line can increase the area
        if height[right] == weak_link:
            right -= 1
        else:
            left += 1

    return max_area
