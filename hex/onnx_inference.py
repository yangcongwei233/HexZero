"""通过 ONNX Runtime 原生内核执行 CPU 推理，无需 PyTorch。"""
import os
from pathlib import Path

import numpy as np

from .model_manifest import load_manifest, model_revision


def default_threads():
    """返回不超过 4 个的默认 CPU 推理线程数；无法识别处理器数量时使用 1 个。"""
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
    def __init__(self, model, kind, threads=None, verify=False):
        """直接加载 ONNX CPU 会话；比赛模式额外检查文件及接口。"""
        import onnxruntime as ort
        self.file_revision = model_revision(model)
        self.verify = verify
        self.manifest = load_manifest(model, kind, verify=verify)
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
        if verify:
            expected = ['board'] if kind == 'trained' else ['board', 'legal_moves']
            if [x.name for x in self.session.get_inputs()] != expected:
                raise ValueError('ONNX 模型输入不匹配')
            if [x.name for x in self.session.get_outputs()] != ['log_policy', 'value']:
                raise ValueError('ONNX 模型输出不匹配')
        self.checkpoint_metadata = self.manifest.get('metadata', {}).copy()

    def eval(self):
        """提供与训练网络一致的评估模式接口；ONNX 会话本身已是推理模式。"""
        return self

    def _board(self, boards, dtype):
        """将输入转成指定类型的连续三维批量棋盘，校验批次非空及棋盘尺寸；单棋盘需由 predict 包装。"""
        boards = np.ascontiguousarray(boards, dtype=dtype)
        n = self.board_size
        if boards.ndim != 3 or boards.shape[1:] != (n, n) or len(boards) == 0:
            raise ValueError('推理棋盘尺寸不匹配')
        return boards

    def predict(self, board):
        """包装单棋盘预测，返回一份策略分布和价值，复用批量推理入口。"""
        policies, values = self.predict_batch([board])
        return policies[0], values[0]

    def predict_batch(self, boards):
        """批量执行本项目模型的策略/价值预测，并将网络输出转换为搜索使用的概率及价值数组。"""
        if self.kind != 'trained':
            raise ValueError('Azalea 推理必须同时提供合法落点')
        boards = self._board(boards, np.float32)
        policy, value = self.session.run(None, {'board': boards})
        return np.exp(policy), value.reshape(-1, 1)

    def run(self, batch):
        """按 Azalea 网络接口接收棋盘和合法动作，调用参考 ONNX 模型并返回命名结果。"""
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
        """加载参考 ONNX 网络及导出搜索设置，检查参考模型棋盘尺寸。"""
        from .azalea_cpu.policy import Policy
        self.checkpoint = Path(checkpoint).resolve()
        net = OnnxNetwork(self.checkpoint, 'azalea', threads)
        if net.board_size != board_size:
            raise ValueError('Azalea 参考模型与训练模型的棋盘大小不同')
        self.board_size = board_size
        self.prototype = Policy(net, net.manifest['search'])

    def player(self, simulations, seed):
        """共享参考网络，为新对局建立独立策略、随机数和参考环境。"""
        import copy
        from .local_players import ReferenceEnvironment, ReferencePlayer
        policy = copy.copy(self.prototype)
        policy.settings = {'move_sampling': False, 'move_exploration': False}
        policy.rng = np.random.RandomState(seed)
        policy.simulations = simulations
        policy.reset()
        return ReferencePlayer(policy, ReferenceEnvironment(self.board_size))
