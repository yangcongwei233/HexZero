"""通过 ONNX Runtime 原生内核执行 CPU 推理，无需 PyTorch。"""
import os
from pathlib import Path

import numpy as np

from .model_manifest import load_manifest


def default_threads():
    return min(4, os.cpu_count() or 1)


def validate_trained_network(net):
    """切换模型前先执行一次真实推理，验证其可用性。"""
    n = net.board_size
    policy, value = net.predict_batch(np.zeros((2, n, n), np.float32))
    if not (policy.shape == (2, n*n) and value.shape == (2, 1)
            and np.isfinite(policy).all() and np.isfinite(value).all()
            and np.all(policy >= 0) and np.allclose(policy.sum(1), 1, atol=1e-5)
            and np.all(np.abs(value) <= 1.00001)):
        raise ValueError('模型试运行输出异常，未切换当前模型')


class OnnxNetwork:
    def __init__(self, model, kind, threads=None):
        import onnxruntime as ort
        self.manifest = load_manifest(model, kind)
        self.board_size = self.manifest['board_size']
        self.kind = kind
        self.threads = default_threads() if threads is None else threads
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or not 1 <= self.threads <= 32:
            raise ValueError('推理线程数必须是 1～32 的整数')
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        expected = ['board'] if kind == 'trained' else ['board', 'legal_moves']
        if [x.name for x in self.session.get_inputs()] != expected:
            raise ValueError('ONNX 模型输入不匹配')
        if [x.name for x in self.session.get_outputs()] != ['log_policy', 'value']:
            raise ValueError('ONNX 模型输出不匹配')
        self.checkpoint_metadata = self.manifest.get('metadata', {}).copy()

    def eval(self):
        return self

    def _board(self, boards, dtype):
        boards = np.ascontiguousarray(boards, dtype=dtype)
        n = self.board_size
        if boards.ndim != 3 or boards.shape[1:] != (n, n) or len(boards) == 0:
            raise ValueError('推理棋盘尺寸不匹配')
        return boards

    def predict(self, board):
        policies, values = self.predict_batch([board])
        return policies[0], values[0]

    def predict_batch(self, boards):
        if self.kind != 'trained':
            raise ValueError('Azalea 推理必须同时提供合法落点')
        boards = self._board(boards, np.float32)
        policy, value = self.session.run(None, {'board': boards})
        return np.exp(policy), value.reshape(-1, 1)

    def run(self, batch):
        if self.kind != 'azalea':
            raise ValueError('此接口仅供 Azalea 搜索使用')
        board = self._board(batch['board'], np.int32)
        legal = np.ascontiguousarray(batch['legal_moves'], dtype=np.int32)
        if legal.ndim != 2 or legal.shape[0] != len(board) or legal.shape[1] == 0:
            raise ValueError('Azalea 合法落点尺寸不匹配')
        policy, value = self.session.run(None, {'board': board, 'legal_moves': legal})
        return {'moves_logprob': policy, 'value': value}


class OnnxReference:
    def __init__(self, checkpoint, board_size, threads=None):
        from .azalea_cpu.policy import Policy
        self.checkpoint = Path(checkpoint).resolve()
        net = OnnxNetwork(self.checkpoint, 'azalea', threads)
        if net.board_size != board_size:
            raise ValueError('Azalea 参考模型与训练模型的棋盘大小不同')
        self.board_size = board_size
        self.prototype = Policy(net, net.manifest['search'])

    def player(self, simulations, seed):
        import copy
        from .local_players import ReferenceEnvironment, ReferencePlayer
        policy = copy.copy(self.prototype)
        policy.settings = {'move_sampling': False, 'move_exploration': False}
        policy.rng = np.random.RandomState(seed)
        policy.simulations = simulations
        policy.reset()
        return ReferencePlayer(policy, ReferenceEnvironment(self.board_size))
