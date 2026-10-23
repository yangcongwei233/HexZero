"""仅用于推理的策略，选步和观察动作逻辑派生自 Azalea 原实现。"""
from __future__ import annotations
import numpy as np
from .search_tree import SearchTree


class Policy:
    supports_time_limit = True
    def __init__(self, net, settings):
        """绑定推理网络并从导出设置读取搜索参数，初始化随机数和采样开关；搜索树由 reset 建立。"""
        self.net = net
        self.settings = {'move_sampling': False, 'move_exploration': False}
        self.rng = np.random.RandomState()
        for name in ('simulations', 'search_batch_size', 'exploration_coef',
                     'exploration_depth', 'exploration_noise_alpha',
                     'exploration_noise_scale', 'exploration_temperature'):
            setattr(self, name, settings[name])

    def reset(self):
        """清空参考搜索树并重置已走步数，开始一局独立对弈。"""
        self.tree = SearchTree()
        self.ply = 0

    def choose_action(self, game: SearchableEnv) \
            -> Tuple[int, Dict[str, Any]]:
        """选择下一步，返回动作和辅助信息；树容量不足时可能抛出 SearchTreeFull。"""
        assert not game.state.result

        temperature = 0.0
        noise_scale = 0.0
        if self.settings['move_sampling']:
            temperature = self.exploration_temperature
            if self.settings['move_exploration']:
                noise_scale = self.exploration_noise_scale
        if self.ply >= self.exploration_depth:
            temperature = 0.

        probs, value, metrics = self.tree.search(
            game, self.net,
            temperature=temperature,
            exploration_noise_scale=noise_scale,
            num_simulations=self.simulations,
            batch_size=self.search_batch_size,
            exploration_coef=self.exploration_coef,
            exploration_noise_alpha=self.exploration_noise_alpha,
            rng=self.rng, time_limit=getattr(self, 'time_limit', 0))
        move_id = np.argmax(self.rng.multinomial(1, probs))
        move = game.state.legal_moves[move_id]
        info = dict(prob=probs[move_id],
                    value=value,
                    moves=game.state.legal_moves,
                    moves_prob=probs,
                    move_id=move_id,
                    metrics=metrics)
        return move, info

    def execute_action(self, move: int, legal_moves: np.ndarray) -> None:
        """根据任一方的落子更新搜索树；容量不足时可能抛出 SearchTreeFull。"""
        move_id = legal_moves.tolist().index(move)
        self.tree.move(move_id)
        self.ply += 1

    def tree_metrics(self):
        """转发底层搜索树的统计接口，供兼容参考策略调用使用。"""
        return self.tree.metrics()

