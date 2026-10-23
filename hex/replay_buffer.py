import os
from collections import deque
from pickle import Pickler, Unpickler
from random import sample


class ReplayBuffer:
    def __init__(self, max_size=0):
        self.max_size = int(max_size or 0)
        self.examples = deque(maxlen=self.max_size if self.max_size > 0 else None)

    def __len__(self):
        return len(self.examples)

    def extend(self, examples):
        self.examples.extend(examples)

    def sample(self, count):
        count = min(int(count), len(self.examples))
        if count <= 0:
            return []
        return sample(list(self.examples), count)

    def to_history(self):
        return [deque(self.examples, maxlen=self.max_size if self.max_size > 0 else None)]

    def save(self, path):
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, "wb+") as f:
            Pickler(f).dump({
                "max_size": self.max_size,
                "examples": list(self.examples),
            })

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = Unpickler(f).load()
        buf = cls(state.get("max_size", 0))
        buf.extend(state.get("examples", []))
        return buf
