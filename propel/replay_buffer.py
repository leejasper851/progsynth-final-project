from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return len(self.buffer)

    def erase(self):
        self.buffer.clear()
