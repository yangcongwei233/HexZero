"""提供紧凑、带版本的教师样本及平衡采样。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FORMAT = 'hexzero-teacher-v1'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_target(game, board, player, legal_moves, visits):
    """Azalea 合法动作使用从 1 开始的原始坐标，区别于网络输入坐标。"""
    legal = np.asarray(legal_moves, dtype=np.int64) - 1
    counts = np.asarray(visits, dtype=np.float64)
    expected = np.flatnonzero(np.asarray(board).ravel() == 0)
    if (not np.array_equal(legal, expected) or counts.shape != legal.shape
            or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0):
        raise ValueError('Teacher visits must match the raw legal-move ordering')
    raw = np.zeros(game.getActionSize(), dtype=np.float32)
    raw[legal] = counts / counts.sum()
    return game.getCanonicalForm(board, player), game.policyToCanonical(raw, player)


def save_dataset(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**metadata, 'format': FORMAT, 'canonical_version': 1,
            'augmentation_version': 2, 'stored_augmentation': 'none',
            'policy_target': 'normalized_raw_mcts_visits', 'value_target': 'terminal_outcome'}
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.teacher-', suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(meta)), **arrays)
        TeacherDataset(temporary, int(meta['board_size']))  # 发布前先校验数据。
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return file_digest(path)


class TeacherDataset:
    def __init__(self, path, board_size, expected_sha256=None):
        self.path = str(Path(path).resolve())
        self.sha256 = file_digest(path)
        if expected_sha256 and expected_sha256 != self.sha256:
            raise ValueError('Teacher dataset hash differs from the saved training configuration')
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata']))
            self.boards = data['boards'].copy()
            self.policies = data['policies'].copy()
            self.values = data['values'].copy()
            self.players = data['players'].copy()
            self.game_ids = data['game_ids'].copy()
            self.plies = data['plies'].copy()
        meta, n, size = self.metadata, int(board_size), len(self.boards)
        if (meta.get('format') != FORMAT or meta.get('board_size') != n
                or meta.get('canonical_version') != 1 or meta.get('augmentation_version') != 2
                or meta.get('stored_augmentation') != 'none'
                or meta.get('policy_target') != 'normalized_raw_mcts_visits'
                or meta.get('value_target') != 'terminal_outcome'):
            raise ValueError('Unsupported teacher dataset format or coordinate convention')
        if (not size or self.boards.shape != (size, n, n) or self.policies.shape != (size, n * n)
                or any(array.shape != (size,) for array in (self.values, self.players, self.game_ids, self.plies))):
            raise ValueError('Invalid teacher dataset array shapes')
        if (not np.isin(self.boards, (-1, 0, 1)).all() or not np.isin(self.values, (-1, 1)).all()
                or not np.isin(self.players, (-1, 1)).all() or not np.isfinite(self.policies).all()
                or (self.policies < 0).any() or not np.allclose(self.policies.sum(1), 1, atol=1e-5)
                or np.any(self.policies[self.boards.reshape(size, -1) != 0] != 0)):
            raise ValueError('Invalid teacher policy, board or outcome labels')
        own = (self.boards == 1).sum((1, 2))
        other = (self.boards == -1).sum((1, 2))
        if (not np.array_equal(other - own, (self.players == -1).astype(int))
                or not np.array_equal(self.plies, own + other + 1)):
            raise ValueError('Teacher color/ply does not agree with the canonical board')
        self.by_color = {player: np.flatnonzero(self.players == player) for player in (1, -1)}
        if any(not len(ids) for ids in self.by_color.values()):
            raise ValueError('Teacher data must contain positions for both colors')

    def __len__(self):
        return len(self.boards)

    def sample(self, count, rng=np.random):
        """先平衡实际棋子颜色，再随机保留原局面或旋转 180°。"""
        count = int(count)
        blue = count // 2 + (int(rng.randint(2)) if count % 2 else 0)
        ids = np.concatenate([rng.choice(self.by_color[1], blue, replace=True),
                              rng.choice(self.by_color[-1], count - blue, replace=True)])
        rng.shuffle(ids)
        result = []
        for index in ids:
            board, policy = self.boards[index], self.policies[index]
            if rng.randint(2):
                board = np.rot90(board, 2).copy()
                policy = policy[::-1].copy()
            result.append((board, policy, float(self.values[index])))
        return result
