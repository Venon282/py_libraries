def display(dct, indent='    ', index=0):
    for key, value in dct.items():
        if isinstance(value, dict):
            print(indent*index, key, ':')
            display(value, indent, index+1)
        else:
            print(indent*index, key, ': ',value)