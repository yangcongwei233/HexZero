"""Inference-only policy; choice/observe logic copied from original Azalea."""
from __future__ import annotations
import numpy as np
from .search_tree import SearchTree


class Policy:
    supports_time_limit = True
    def __init__(self, net, settings):
        self.net = net
        self.settings = {'move_sampling': False, 'move_exploration': False}
        self.rng = np.random.RandomState()
        for name in ('simulations', 'search_batch_size', 'exploration_coef',
                     'exploration_depth', 'exploration_noise_alpha',
                     'exploration_noise_scale', 'exploration_temperature'):
            setattr(self, name, settings[name])

    def reset(self):
        self.tree = SearchTree()
        self.ply = 0

    def choose_action(self, game: SearchableEnv) \
            -> Tuple[int, Dict[str, Any]]:
        """Choose next move.
        can raise SearchTreeFull
        :param game: Current game environment
        :returns: move - chosen move
                  info - auxiliary information
        """
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
        """Update search tree with own or opponent action.
        can raise SearchTreeFull
        """
        move_id = legal_moves.tolist().index(move)
        self.tree.move(move_id)
        self.ply += 1

    def tree_metrics(self):
        return self.tree.metrics()

