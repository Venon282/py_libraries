#! Better to use the library collections
class Stack:
    def __init__(self):
        """
        Initializes an empty Stack.
        """
        self.stack = []

    def push(self, element):
        self.stack.append(element)

    def pop(self):
        return self.stack.pop()

    def empty(self):
        return len(self.stack) == 0

    def top(self):
        """return the reference of the top value

        Returns:
            _type_: _description_
        """
        return self.stack[-1]

    def min(self):
        return min(self.stack)

    def max(self):
        return max(self.stack)
