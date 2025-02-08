class Queue:
    def __init__(self):
        """
        Initializes an empty Stack.
        """
        self.queue = []

    def enqueue(self, element):
        self.queue = [element] + self.queue

    def dequeue(self):
        return self.queue.pop()

    def empty(self):
        return len(self.queue) == 0

    def top(self):
        return self.queue[-1]

    def min(self):
        return min(self.queue)

    def max(self):
        return max(self.queue)
