class NeuralNet():
    """策略/价值网络接口，输入统一为当前执棋方视角的棋盘。"""

    def __init__(self, game):
        pass

    def train(self, examples):
        """使用自博弈样本训练；每条样本为规范化棋盘、搜索策略 pi 和价值 v。"""
        pass

    def predict(self, board):
        """预测规范化棋盘，返回动作概率向量 pi 和范围为 [−1, 1] 的价值 v。"""
        pass

    def save_checkpoint(self, folder, filename):
        """将网络参数保存到 folder/filename。"""
        pass

    def load_checkpoint(self, folder, filename):
        """从 folder/filename 加载网络参数。"""
        pass
