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