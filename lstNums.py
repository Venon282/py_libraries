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
