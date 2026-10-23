import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


def arg_value(args, name, default):
    """读取搜索配置，缺失字段使用给定默认值。"""
    try:
        return getattr(args, name)
    except (AttributeError, KeyError):
        return default


class MCTS():
    """维护逐次模拟的 MCTS 搜索树。"""

    def __init__(self, game, nnet, args):
        """绑定游戏和网络，初始化局面、边访问、价值及策略缓存，并建立独立随机数生成器。"""
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # 状态与动作对应的平均价值 Q。
        self.Nsa = {}  # 状态与动作对应的边访问次数。
        self.Ns = {}  # 局面访问次数。
        self.Ps = {}  # 网络输出的初始策略。

        self.Es = {}  # 缓存局面的终局结果。
        self.Vs = {}  # 缓存局面的合法动作。
        self.rng = np.random.default_rng(arg_value(args, 'seed', None))

    def getActionProb(self, canonicalBoard, temp=1):
        """从规范化棋盘执行指定次数的模拟。
        返回策略分布，动作概率正比于访问次数的 1/temp 次方。"""
        if self.args.numMCTSSims < 1 or temp < 0:
            raise ValueError('numMCTSSims must be positive and temperature nonnegative')
        if self.game.getGameEnded(canonicalBoard, 1):
            raise ValueError('Cannot search a terminal board')
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())], dtype=np.float64)
        counts *= valids
        if counts.sum() == 0:
            counts = self.Ps[s].copy()

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = self.rng.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = (counts / counts.max()) ** (1. / temp)
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """执行一次模拟，递归选择置信上界评分最高的动作直至叶节点。
        非终局叶子由网络给出策略和价值，终局叶子直接使用胜负结果。
        沿路径更新 Ns、Nsa、Qsa；返回值取反，转换为上一层对手视角。"""

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # 终局节点。
            return -self.Es[s]

        if s not in self.Ps:
            # 尚未展开的叶节点。
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            heuristic_prior_weight = max(0.0, min(1.0, arg_value(self.args, 'heuristicPriorWeight', 0.0)))
            if heuristic_prior_weight > 0:
                from hex.hex_heuristics import heuristic_policy
                heuristic = heuristic_policy(canonicalBoard, 1, self.game)
                self.Ps[s] = (1.0 - heuristic_prior_weight) * self.Ps[s] + heuristic_prior_weight * heuristic
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # 屏蔽非法动作。
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # 重新归一化。
            else:
                # 合法动作概率全部为零时，改用均匀分布。

                # 合法概率全为零可能与网络能力不足或过拟合有关。
                # 频繁出现此提示时，应检查网络和训练过程。
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = valids.astype(float)
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # 选择置信上界评分最高的动作。
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # 未访问边的价值按零处理。

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
