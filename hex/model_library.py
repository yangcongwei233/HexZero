"""发现可用于本地对弈的 HexZero ONNX 模型，无需导入 PyTorch。"""
import json
from pathlib import Path

from .model_manifest import load_manifest


def check_compatible(manifest, board_size=11):
    if manifest.get('board_size') != board_size:
        raise ValueError(f'只支持 {board_size}×{board_size} 模型')
    if manifest.get('canonical_version') != 1 or manifest.get('precision') != 'FP32':
        raise ValueError('需要 canonical_version=1 的 FP32 HexZero 导出模型')
    if not isinstance(manifest.get('metadata', {}), dict):
        raise ValueError('模型元数据格式无效')


class ModelLibrary:
    def __init__(self, root, board_size=11):
        self.root = Path(root).resolve()
        self.board_size = board_size
        self.cache = {}
        self.entries = []
        self.refresh()

    def model_id(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def refresh(self):
        paths = set(self.root.rglob('*.onnx')) if self.root.is_dir() else set()
        if self.root.is_dir():
            paths.update(Path(str(p)[:-5]) for p in self.root.rglob('*.onnx.json'))
        entries, cache = [], {}
        for path in sorted(paths):
            if not path.resolve().is_relative_to(self.root):
                continue
            ident = path.relative_to(self.root).as_posix()
            manifest_path = Path(str(path)+'.json')
            def file_signature(file):
                try:
                    info = file.stat()
                    return info.st_size, info.st_mtime_ns
                except OSError:
                    return None  # 复制模型期间，目录可能被移动。
            signature = tuple(file_signature(p) for p in (path, manifest_path))
            previous = self.cache.get(ident)
            if previous and previous[0] == signature:
                entry = previous[1]
            else:
                entry = dict(id=ident, label=ident, available=False)
                try:
                    if not path.is_file():
                        raise ValueError('等待复制 ONNX 文件')
                    if not manifest_path.is_file():
                        raise ValueError('缺少同名 .onnx.json 清单，请一起复制')
                    header = json.loads(manifest_path.read_text(encoding='utf-8'))
                    if isinstance(header, dict) and header.get('kind') == 'azalea':
                        entry = None  # 参考模型有独立的固定用途。
                    else:
                        manifest = load_manifest(path, 'trained')
                        check_compatible(manifest, self.board_size)
                        meta = manifest.get('metadata', {})
                        iteration = meta.get('iteration')
                        iteration = iteration if type(iteration) is int and iteration >= 0 else None
                        run = meta.get('run_config', {})
                        fraction = run.get('teacherFraction', 0) if isinstance(run, dict) else 0
                        fraction = fraction if type(fraction) in (int, float) and 0 <= fraction <= 1 else 0
                        title = f'第 {iteration} 轮' if iteration is not None else path.stem
                        if fraction: title += f' · 教师 {fraction:.0%}'
                        entry.update(available=True, iteration=iteration, teacherFraction=fraction,
                                     sha256=manifest['onnx_sha256'], label=f'{title} · {ident}')
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    entry['error'] = str(exc)
            cache[ident] = signature, entry
            if entry is not None: entries.append(entry)
        # 提示未经转换的 PyTorch 断点为何不可直接选择。
        if self.root.is_dir():
            for path in sorted(self.root.rglob('*')):
                if (path.is_file() and path.name.endswith(('.pth', '.pth.tar', '.pt'))
                        and path.resolve().is_relative_to(self.root)
                        and not any(p.parent == path.parent for p in paths)):
                    ident = path.relative_to(self.root).as_posix()
                    entries.append(dict(id=ident, label=ident, available=False,
                                        error='这是 PyTorch 权重；请先导出 ONNX，并复制 .onnx 与 .onnx.json'))
        entries.sort(key=lambda e: (not e['available'], -(e.get('iteration') or 0), e['id']))
        self.entries, self.cache = entries, cache
        return self.snapshot()

    def snapshot(self):
        return dict(directory=str(self.root), entries=[dict(e) for e in self.entries], refreshSeconds=5)

    def resolve(self, ident):
        if not isinstance(ident, str): raise ValueError('请选择模型')
        self.refresh()
        entry = next((e for e in self.entries if e['id'] == ident), None)
        if entry is None: raise ValueError('模型已移除或不在 models 文件夹中，请刷新列表')
        if not entry['available']: raise ValueError(entry['error'])
        path = (self.root / ident).resolve()
        if not path.is_relative_to(self.root): raise ValueError('模型路径超出 models 文件夹')
        # 选择时重新计算哈希，即使文件时间戳未变化。
        check_compatible(load_manifest(path, 'trained'), self.board_size)
        return path

    def choose(self, requested=None):
        self.refresh()
        valid = [e for e in self.entries if e['available']]
        if not valid: raise ValueError('models 中没有可用模型；请复制完整的 .onnx 和 .onnx.json')
        if requested is None: return valid[0]['id']
        requested = str(requested)
        if requested.isdigit():
            preferred = f'iteration_{requested}/trained.onnx'
            if any(e['id'] == preferred for e in valid): return preferred
            matches = [e for e in valid if e['iteration'] == int(requested)]
            if len(matches) == 1: return matches[0]['id']
            if len(matches) > 1: raise ValueError('该轮次有多个模型，请使用模型的相对路径')
        self.resolve(requested)
        return requested
