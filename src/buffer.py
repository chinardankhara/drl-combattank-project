import random
from collections import deque, namedtuple


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory = deque([], maxlen = self.capacity)
        return self
