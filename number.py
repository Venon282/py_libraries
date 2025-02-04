def nextMultipleOf(n, multiple):
    return n if n % multiple == 0 else (n // multiple + 1) * multiple

def isPalindrome(x):
    """
       :type x: int
       :rtype: bool
    """
    x_str = str(x)
    return x_str == x_str[::-1]

def maxAscendingSum(nums):
    """
        :type nums: List[int]
        :rtype: int
    """
    count = max_  = nums[0]
    for i in range(1, len(nums)):
        if nums[i-1] < nums[i]:
            count += nums[i]
        else:
            max_ = max(max_, count)
            count = nums[i]
    return max(max_, count)
