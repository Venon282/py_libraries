import numpy as np

def getNextMultipleOf(n: int, multiple: int) -> int:
    """Return the next multiple of a number greater than or equal to n.

    Args:
        n: The number to find the next multiple for.
        multiple: The multiple to align to.

    Returns:
        The smallest multiple of `multiple` that is greater than or equal to n.
    """
    return n if n % multiple == 0 else (n // multiple + 1) * multiple

def isPalindrome(x: int) -> bool:
    """Check if a number is a palindrome.

    Args:
        x: The number to check.

    Returns:
        True if the number is a palindrome, False otherwise.
    """
    x_str = str(x)
    return x_str == x_str[::-1]

def computeMaxAscendingSum(nums: list[int]) -> int:
    """Compute the maximum sum of an ascending subarray.

    Args:
        nums: List of integers to process.

    Returns:
        The maximum sum of any contiguous ascending subarray.
    """
    current_sum = max_sum = nums[0]
    for i in range(1, len(nums)):
        if nums[i - 1] < nums[i]:
            current_sum += nums[i]
        else:
            max_sum = max(max_sum, current_sum)
            current_sum = nums[i]
    return max(max_sum, current_sum)

roman = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}

def romanToInt(s: str) -> int:
    """Convert a Roman numeral string to an integer.

    Args:
        s: Roman numeral string to convert.

    Returns:
        The integer value of the Roman numeral.
    """
    num = i = 0
    while i < len(s):
        if i < len(s) - 1 and roman[s[i + 1]] > roman[s[i]]:
            num += roman[s[i + 1]] - roman[s[i]]
            i += 2
        else:
            num += roman[s[i]]
            i += 1

    return num

def intToRoman(num: int) -> str:
    roman_pairs  = [('M', 1000), ('CM', 900), ('D', 500), ('CD', 400),
                    ('C', 100), ('XC', 90), ('L', 50), ('XL', 40),
                    ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1)]
    res = ''
    
    for k, n in roman_pairs:
        while num >= n:
            res += k
            num -= n
        
    return res

def isPowerOfTwo(n: int) -> bool:
    """Check if a number is a power of two.

    Args:
        n: The number to check.

    Returns:
        True if the number is a power of two, False otherwise.
    """
    return n > 0 and (n & (n - 1)) == 0

def computeDigitalRoot(num: int) -> int:
    """Compute the digital root of a number (repeated sum of digits until single digit).

    Args:
        num: The number to compute the digital root for.

    Returns:
        The digital root of the number.

    Example:
        Input: num = 38
        Output: 2
        Explanation: The process is 38 -> 3 + 8 -> 11, 11 -> 1 + 1 -> 2.
        Since 2 has only one digit, return it.
    """
    return 0 if num == 0 else (num % 9 or 9)
def getSign(n: int | float) -> int:
    """Get the sign of a number.

    Args:
        n: The number to get the sign for.

    Returns:
        1 if positive, -1 if negative, 0 if zero.
    """
    return (n > 0) - (n < 0)

def reverseNumber(n: int) -> int:
    """Reverse the digits of a number.

    Args:
        n: The number to reverse.

    Returns:
        The number with its digits reversed.
    """
    return int(str(abs(n))[::-1]) * getSign(n)

def formatNumber(num: any, boundMin: float = 0.001, boundMax: float = 1e6, precision: int = 3) -> str:
    """
    Returns a human-readable string representation of a number.

    - If num is 0, returns "0".
    - For numbers in [boundMin, boundMax), uses fixed-point notation.
      Trailing zeros in the fractional part are omitted.
    - For numbers < boundMin or >= boundMax, uses scientific notation.
      The mantissa is trimmed of trailing zeros, and the exponent is
      shown in a compact form.

    Examples:
      formatNumber(0.0000000256)   -> "2.56e-8"
      formatNumber(0.654212574854)  -> "0.65"
      formatNumber(455351356)       -> "4.55e8"
      formatNumber(1.0)             -> "1"
    
    Args:
      num (float): The number to format.
      boundMin (float, optional): Lower bound for fixed-point formatting.
      boundMax (float, optional): Upper bound for fixed-point formatting.
      precision (int, optional): Maximum number of decimal places.
    
    Returns:
      str: The formatted number.
    """
    try:
        num = float(num)
    except (TypeError, ValueError):
        return str(num)

    def trimTrailingZeros(s: str) -> str:
        # Remove trailing zeros and an extraneous decimal point, if any.
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    # Special case for zero.
    if num == 0:
        return "0"
    
    abs_num = abs(num)
    
    if abs_num < boundMin or abs_num >= boundMax:
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

def generateRandomGaussianWithBounds(boundMin: float | int, boundMax: float | int, mean: float | int | None = None, std: float | int | None = None, retry: bool = True, dispersion: int = 3) -> float:
    """
    Generate a random number from a Gaussian distribution, 
    then clamp it within [boundMin, boundMax].

    Parameters
    ----------
    boundMin : float or int
        Lower bound of the allowed range.
    boundMax : float or int
        Upper bound of the allowed range.
    mean : float or int, optional
        Mean of the Gaussian distribution. Default is midpoint of bounds.
    std : float or int, optional
        Standard deviation of the Gaussian distribution. Default is (mean - boundMin) / 3. 99.7% fall within ±3 standard deviations.
    retry : bool, optional
        If True, will retry generating a number if it falls outside the bounds. Default is False.
    dispersion : int, optional 
        Factor to determine default std if not provided. Default is 3. More dispersion means a smaller std.
    
    Returns
    -------
    float
        Random value drawn from the Gaussian distribution, 
        clamped within [boundMin, boundMax].
    """
    if mean is None:
        mean = (boundMin + boundMax) / 2
    if std is None:
        std = (mean - boundMin) / dispersion

    rand = np.random.normal(loc=mean, scale=std)
    
    while retry and (rand < boundMin or rand > boundMax):
        rand = np.random.normal(loc=mean, scale=std)
        
    return min(max(boundMin, rand), boundMax)