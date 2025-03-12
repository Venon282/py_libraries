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

def addDigitsUntilOnlyOne(num):
        """
        :type num: int
        :rtype: int
        
        Exemple:
            Input: num = 38
            Output: 2
            Explanation: The process is
            38 --> 3 + 8 --> 11
            11 --> 1 + 1 --> 2 
            Since 2 has only one digit, return it.
        """
        return 0 if num == 0 else (num % 9 or 9)
def sign(n):
    return (n > 0) - (n < 0)

def reverse(n):
    return int(str(abs(n))[::-1]) * sign(n)

def formated(num: float, bound_min: float = 0.001, bound_max: float = 1e6, precision: int = 3) -> str:
    """
    Returns a human-readable string representation of a number.

    - If num is 0, returns "0".
    - For numbers in [bound_min, bound_max), uses fixed-point notation.
      Trailing zeros in the fractional part are omitted.
    - For numbers < bound_min or >= bound_max, uses scientific notation.
      The mantissa is trimmed of trailing zeros, and the exponent is
      shown in a compact form.

    Examples:
      formated(0.0000000256)   -> "2.56e-8"
      formated(0.654212574854)  -> "0.65"
      formated(455351356)       -> "4.55e8"
      formated(1.0)             -> "1"
    
    Args:
      num (float): The number to format.
      bound_min (float, optional): Lower bound for fixed-point formatting.
      bound_max (float, optional): Upper bound for fixed-point formatting.
      precision (int, optional): Maximum number of decimal places.
    
    Returns:
      str: The formatted number.
    """

    def trimTrailingZeros(s: str) -> str:
        # Remove trailing zeros and an extraneous decimal point, if any.
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    # Special case for zero.
    if num == 0:
        return "0"
    
    abs_num = abs(num)
    
    if abs_num < bound_min or abs_num >= bound_max:
        # Use scientific notation.
        
        s = f"{num:.{precision}e}"  # e.g. "4.50e+08" or "2.56e-08"
        if 'e' in s:
            mantissa, exponent = s.split("e")
            mantissa = trimTrailingZeros(mantissa)
            # Convert exponent to integer to remove unnecessary '+' and leading zeros.
            exp_int = int(exponent)
            
            return f"{mantissa}e{exp_int}"
        else:
            return s
    else:
        # Use fixed-point notation.
        s = f"{num:.{precision}f}"  # e.g. "1.00", "0.65"
        return trimTrailingZeros(s)

