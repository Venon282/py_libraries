def nextMultipleOf(n, multiple):
    return n if n % multiple == 0 else (n // multiple + 1) * multiple

def isPalindrome(self, x):
    """
       :type x: int
       :rtype: bool
    """
    x_str = str(x)
    return x_str == x_str[::-1]
