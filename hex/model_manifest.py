"""Read model manifests without importing any native inference library."""
import hashlib
import json
from pathlib import Path


def load_manifest(model, kind):
    model = Path(model)
    data = json.loads(model.with_suffix(model.suffix + '.json').read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('format') != 'hexzero-onnx-v1' or data.get('kind') != kind:
        raise ValueError(f'ONNX 模型类型或格式不匹配：{model.name}')
    if hashlib.sha256(model.read_bytes()).hexdigest() != data.get('onnx_sha256'):
        raise ValueError(f'ONNX 模型校验失败，请重新解压：{model.name}')
    n = data.get('board_size')
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 25:
        raise ValueError('ONNX 模型缺少有效棋盘尺寸')
    return data

