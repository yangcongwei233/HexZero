#!/usr/bin/env python3
"""导出本地对弈网络，校验后成对发布模型及清单。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HEX_FORCE_CPU'] = '1'

import numpy as np
import onnx
import onnxruntime as ort
import torch

from hex.HexGame import HexGame
from hex.local_players import AzaleaReference, ReferenceEnvironment
from hex.pytorch.NNet import NNetWrapper


def sha(path):
    """计算源权重或导出模型的哈希，用于清单中的来源与内容校验。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceExport(torch.nn.Module):
    def __init__(self, net):
        """包装参考网络，保留原前向计算而统一导出返回结构。"""
        super().__init__()
        self.net = net

    def forward(self, board, legal_moves):
        """将参考网络的命名字典输出转换为策略对数概率和价值元组，供 ONNX 导出使用。"""
        result = self.net(board, legal_moves)
        return result['moves_logprob'], result['value']


def validation_inputs(size, kind):
    """用固定种子的合法落子构造不同阶段棋盘及参考动作数组，作为导出前后数值验证输入。"""
    game = HexGame(size)
    board, player = game.getInitBoard(), 1
    rng = np.random.default_rng(20261015)
    boards, moves = [], []
    for _ in range(64):
        if kind == 'trained':
            boards.append(game.getCanonicalForm(board, player).astype(np.float32))
        else:
            b = np.where(board == -1, 2, board).astype(np.int32)
            legal = (np.flatnonzero(board.ravel() == 0) + 1).astype(np.int32)
            if player == -1:
                b, legal = ReferenceEnvironment.flip_player_board_moves(b, legal)
                b, legal = b[0], legal[0]
            boards.append(b)
            moves.append(legal)
        action = int(rng.choice(np.flatnonzero(board.ravel() == 0)))
        board, player = game.getNextState(board, player, action)
        if game.getGameEnded(board, player):
            board, player = game.getInitBoard(), 1
    return np.asarray(boards), moves


def export(model, path, info):
    """导出临时 ONNX，在实际会话中核对输出形状和数值误差，验证通过后发布模型与带哈希清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    kind, n = info['kind'], info['board_size']
    boards, legal = validation_inputs(n, kind)
    inputs = (torch.from_numpy(boards[:3]),)
    names = ['board']
    axes = dict(board={0: 'batch'}, log_policy={0: 'batch'}, value={0: 'batch'})
    if kind == 'azalea':
        inputs += (torch.arange(1, n*n+1, dtype=torch.int32).repeat(3, 1),)
        names.append('legal_moves')
        axes['legal_moves'] = {0: 'batch', 1: 'moves'}
        axes['log_policy'][1] = 'moves'
    with tempfile.TemporaryDirectory(prefix='onnx-export-', dir=path.parent) as folder:
        temporary = Path(folder) / path.name
        with torch.inference_mode():
            torch.onnx.export(model.eval(), inputs, str(temporary), input_names=names,
                              output_names=['log_policy', 'value'], dynamic_axes=axes,
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(str(temporary))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
        validation = []
        with torch.inference_mode():
            for count in (1, 3, 10, 32):
                for start in (0, 32):
                    feed = {'board': boards[start:start+count]}
                    if kind == 'azalea':
                        choices = legal[start:start+count]
                        padded = np.zeros((count, max(map(len, choices))), dtype=np.int32)
                        for i, row in enumerate(choices):
                            padded[i, :len(row)] = row
                        feed['legal_moves'] = padded
                    expected = model(*(torch.from_numpy(feed[name]) for name in names))
                    actual = session.run(None, feed)
                    p0, p1 = np.exp(expected[0].numpy()), np.exp(actual[0])
                    v0, v1 = expected[1].numpy(), actual[1]
                    np.testing.assert_allclose(p0, p1, atol=2e-5, rtol=3e-4)
                    np.testing.assert_allclose(v0, v1, atol=2e-5, rtol=3e-4)
                    validation.append(dict(batch=count, start=start,
                                           policy_max_abs=float(np.max(np.abs(p0-p1))),
                                           value_max_abs=float(np.max(np.abs(v0-v1)))))
        info.update(format='hexzero-onnx-v1', precision='FP32', onnx_sha256=sha(temporary),
                    exporter=dict(torch=torch.__version__, onnx=onnx.__version__,
                                  onnxruntime=ort.__version__, opset=17), validation=validation)
        manifest = Path(folder) / (path.name + '.json')
        manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        del session  # 替换文件前释放模型，也适用于 Windows。
        temporary.replace(path)
        manifest.replace(path.with_suffix(path.suffix + '.json'))
    print(json.dumps(dict(model=str(path), bytes=path.stat().st_size, validation='passed')), flush=True)


def export_trained(checkpoint, destination):
    """校验本项目断点坐标版本，按保存的网络结构加载权重，调用统一导出与验证流程。"""
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('canonical_version') != 1:
        raise ValueError('只支持 canonical_version=1 的 HexZero 断点')
    game = HexGame(int(state['board_size'][0]))
    net = NNetWrapper(game, {'cuda': False})
    net.load_training_state({k: v for k, v in state.items() if k not in ('optimizer_state_dict', 'rng_state')})
    metadata = state.get('metadata', {}).copy()
    sidecar = checkpoint.with_suffix(checkpoint.suffix+'.json')
    if sidecar.is_file():
        extra = json.loads(sidecar.read_text(encoding='utf-8'))
        if extra.get('sha256') and extra['sha256'] != sha(checkpoint):
            raise ValueError('断点与其 JSON 来源清单校验值不一致')
        metadata.update(extra)
    export(net.nnet, destination,
           dict(kind='trained', board_size=game.n, metadata=metadata,
                source_checkpoint=checkpoint.name, source_sha256=sha(checkpoint),
                network_config=state['network_config'], canonical_version=state['canonical_version']))


def main():
    """解析输出位置及模型选择，分别导出本项目模型或可选 Azalea 参考模型，保存各自推理元数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/local_play/onnx')
    parser.add_argument('--reference-source', type=Path, default=ROOT.parent / 'azalea')
    parser.add_argument('--checkpoint', type=Path,
                        help='Export any single HexZero checkpoint to OUTPUT/trained.onnx and its JSON manifest')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.checkpoint:
        export_trained(args.checkpoint, args.output / 'trained.onnx')
        return
    for iteration in (750, 800):
        checkpoint = ROOT / f'models/local_play/archive/iteration_{iteration}/trained.pth.tar'
        export_trained(checkpoint, args.output / f'iteration_{iteration}/trained.onnx')
    checkpoint = args.reference_source / 'models/hex11-20180712-3362.policy.pth'
    reference = AzaleaReference(args.reference_source, checkpoint, 11, 'cpu')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['policy']
    names = ('simulations', 'search_batch_size', 'exploration_coef', 'exploration_depth',
             'exploration_noise_alpha', 'exploration_noise_scale', 'exploration_temperature')
    export(ReferenceExport(reference.prototype.net), args.output / 'azalea/reference.onnx',
           dict(kind='azalea', board_size=11, source_checkpoint=checkpoint.name,
                source_sha256=sha(checkpoint), search={name: state[name] for name in names}))


if __name__ == '__main__':
    main()
