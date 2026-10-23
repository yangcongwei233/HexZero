class AverageMeter(object):
    """参考实现：https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        """初始化当前值、加权总和、计数及平均值，用于汇总批次损失。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        """用科学计数法展示当前平均值，供进度条和日志使用。"""
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        """用 n 个样本的均值更新加权累计量，刷新总平均值。"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        """将属性读取映射为字典键读取，缺失键抛出 AttributeError 以兼容 getattr 默认值。"""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__
