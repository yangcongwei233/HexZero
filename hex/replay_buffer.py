import os
from collections import deque
from pickle import Pickler, Unpickler
from random import sample


class ReplayBuffer:
    def __init__(self, max_size=0):
        """创建 FIFO 经验队列，正容量限制样本数，零表示不设容量上限。"""
        self.max_size = int(max_size or 0)
        self.examples = deque(maxlen=self.max_size if self.max_size > 0 else None)

    def __len__(self):
        """返回当前缓存样本数，供训练安排批次和检查空池。"""
        return len(self.examples)

    def extend(self, examples):
        """追加新样本，达到容量上限时由队列自动淘汰最旧数据。"""
        self.examples.extend(examples)

    def sample(self, count):
        """从现存经验中无放回抽取最多 count 条，空池或非正请求返回空列表。"""
        count = min(int(count), len(self.examples))
        if count <= 0:
            return []
        return sample(list(self.examples), count)

    def to_history(self):
        """转换为训练器使用的单段历史形式，保持当前样本顺序及容量约束。"""
        return [deque(self.examples, maxlen=self.max_size if self.max_size > 0 else None)]

    def save(self, path):
        """将容量和当前样本写入指定文件，必要时创建父目录。"""
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
        """从缓存文件重建队列和容量，按原顺序恢复样本。"""
        with open(path, "rb") as f:
            state = Unpickler(f).load()
        buf = cls(state.get("max_size", 0))
        buf.extend(state.get("examples", []))
        return buf
