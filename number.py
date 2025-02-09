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

roman = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}

def romanToInt(self, s):
    """
        :type s: str
        :rtype: int
    """
    num = i = 0
    while i < len(s):
        if i < len(s)-1 and roman[s[i+1]] > roman[s[i]]:
            num += roman[s[i+1]] - roman[s[i]]
            i+=2
        else:
            num += roman[s[i]]
            i+=1

    return num

def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0
