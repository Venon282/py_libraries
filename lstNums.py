from typing import List

def tupleSameProduct(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    dict_ = defaultdict(int)
    ans = 0

    for i, num in enumerate(nums[:-1]):
        for n in nums[i+1:]:
            product = num * n
            ans += 8 * dict_[product]
            dict_[product] +=1
    return ans

def idxSumOfTwo(nums, target):
    """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
    """
    dict_ = {}
    for i in range(len(nums)):
        if nums[i] in dict_:
            return [dict_[nums[i]], i]
        dict_[target - nums[i]] = i
    return []

def mergeIntervals(intervals):
    """
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    """
    intervals = sorted(intervals, key=lambda x: x[0])
    start, end = intervals.pop(0)
    new = [[start, end]]

    for start, end in intervals:
        if new[-1][1] >= start:
            new[-1][1] = max(new[-1][1], end)
        else:
            new.append([start, end])
    return new

def oddNumberInEvenList(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    x=0
    for i in nums:
        x^=i
    return x

def pascalTriangle(n_rows):
    triangle = [[1]]
    for _ in range(n_rows - 1):
        next_row = [1] + [a + b for a, b in zip(triangle[-1], triangle[-1][1:])] + [1]
        triangle.append(next_row)
    return triangle

def minQueriesForZeroArray(nums:List[int], queries:List[int]) -> int:
    """
    Determines the minimum number of queries required to transform the array `nums`
    into a zero array. Each query is represented as [left, right, val], and for each query,
    every element in the range [left, right] can be decreased by at most `val`.

    Parameters:
        nums (List[int]): The initial array of integers.
        queries (List[List[int]]): A list of queries, where each query is in the form [left, right, val].

    Returns:
        int: The minimum number of queries needed to turn `nums` into a zero array.
             Returns -1 if it is not possible using the given queries.
    """
    
    # Initialize a difference array to efficiently apply range updates.
    # Its size is len(nums) + 1 to handle the range update end offset.
    diffs = [0] * (len(nums) + 1)
    
    query_idx = 0  # Counter for the number of queries applied.
    total_sum = 0  # Cumulative sum representing the current update applied to nums.
    
    # Iterate through each index of the array nums.
    for i in range(len(nums)):
        # While the cumulative sum at index i (including the difference update)
        # is less than the target value in nums, apply additional queries.
        while total_sum + diffs[i] < nums[i]:
            query_idx += 1  # Use the next query.
            
            # If we run out of queries, return -1 as it's not possible to form a zero array.
            if query_idx > len(queries):
                return -1
            
            # Retrieve the current query parameters.
            left, right, val = queries[query_idx - 1]
            
            # If the query affects the current index i, update the difference array.
            if right >= i:
                # Start applying the query from the maximum of left or the current index,
                # so we only update the elements from i onwards.
                diffs[max(left, i)] += val
                # End the update effect right after the query's right boundary.
                diffs[right + 1] -= val
        
        # Update the cumulative sum for index i.
        total_sum += diffs[i]
    
    # Return the total number of queries applied to achieve a zero array.
    return query_idx

def maxItemsPerPerson(piles, k):
    """
    Calculates the maximum number of items that can be allocated to each person 
    from a list of piles, given that:
      - Each element in 'piles' represents the number of items in that pile.
      - Each pile can be divided into sub-piles, but items from different piles 
        cannot be merged.
      - There are 'k' persons, and each person must receive items exclusively from a single pile.
    
    Parameters:
    piles (List[int]): A list of integers where each integer represents the size of a pile.
    k (int): The number of persons to allocate items to.
    
    Returns:
    int: The maximum number of items each person can receive. Returns 0 if the total 
         number of items is less than k.
    """
    # If there aren't enough items to allocate at least one item per person, return 0.
    if sum(piles) < k:
        return 0
    
    # Define binary search bounds:
    # Lower bound: 1 (minimum allocation per person)
    # Upper bound: the maximum number of items available in a single pile.
    left, right = 1, max(piles)

    # Perform binary search to determine the maximum valid allocation per person.
    while left <= right:
        # Candidate allocation for each person.
        middle = left + (right - left) // 2
        
        # Calculate total number of persons that can be allocated 'middle' items.
        # Each pile contributes floor(pile / middle) persons.
        if sum(pile // middle for pile in piles) >= k:
            # If it's possible to allocate 'middle' items per person, try a larger allocation.
            left = middle + 1
        else:
            # Otherwise, reduce the allocation candidate.
            right = middle - 1

    # 'right' holds the maximum number of items each person can receive.
    return right
