class Game():
    """双人轮流对抗游戏的规则接口，子类需实现下列方法。
    双方分别用 1 和 −1 表示。"""
    def __init__(self):
        pass

    def getInitBoard(self):
        """返回初始棋盘，表示形式应便于输入网络。"""
        pass

    def getBoardSize(self):
        """返回棋盘的行数和列数。"""
        pass

    def getActionSize(self):
        """返回全部可能动作的数量。"""
        pass

    def getNextState(self, board, player, action):
        """在 board 上由 player 执行动作 action，返回新棋盘及下一方。"""
        pass

    def getValidMoves(self, board, player):
        """返回当前方的合法动作掩码，长度为动作总数，1 表示合法。"""
        pass

    def getGameEnded(self, board, player):
        """返回当前方的结果：未结束为 0，胜为 1，负为 −1。
        支持平局的游戏应返回约定的非零平局值。"""
        pass

    def getCanonicalForm(self, board, player):
        """将棋盘转换为当前执棋方的统一视角，供网络和搜索使用。"""
        pass

    def getSymmetries(self, board, pi):
        """返回棋盘及策略向量的合法对称变换，用于训练数据增强。"""
        pass

    def stringRepresentation(self, board):
        """返回棋盘的紧凑表示，供 MCTS 作为局面键。"""
        pass
