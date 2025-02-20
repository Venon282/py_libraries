def lackingBinaries(lst, length=None):
    """
    Return the binaries number that are not present in the list at the defined length
    Default length is the lenght of the binary max
    """
    if length is None:
        length = len(max(lst)) - 2
    return [bin(i) for i in range(2**length) if bin(i) not in lst]
    
def binaryToDecimal(binary):
    return int(binary, 2)

def decimalToBinary(decimal):
    return bin(decimal)