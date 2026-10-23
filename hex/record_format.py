"""比赛棋谱 TXT：GB2312 编码，A1 位于左下，红方先手连接上下。"""
import json
import math
import re

from .HexGame import HexGame

LABELS = {-2: '大劣', -1: '小劣', 0: '一般', 1: '小好', 2: '大好'}
RESULTS = ('先手胜', '后手胜')
MAX_RECORD_BYTES = 262144
MAX_JSON_RECORD_BYTES = 8 * 1024 * 1024


def coordinate(action, size=11):
    row, col = divmod(int(action), size)
    return f'{chr(65 + col)},{size - row}'


def mark_from_value(value):
    if not math.isfinite(value) or not -1.00001 <= value <= 1.00001:
        raise ValueError('模型估值无效')
    return -2 if value < -.6 else -1 if value < -.2 else 0 if value <= .2 else 1 if value <= .6 else 2


def metadata_checked(meta):
    if not isinstance(meta, dict):
        raise ValueError('参赛信息必须是对象')
    result = {}
    for name, default in (('redTeam', '先手队'), ('blueTeam', '后手队'),
                          ('result', ''), ('timePlace', ''), ('event', '本地对弈')):
        text = meta.get(name, default)
        if not isinstance(text, str) or len(text) > 160:
            raise ValueError('参赛信息必须是 160 字以内的文本')
        text = ' '.join(text.split())
        if any(c in text for c in '[]{};'):
            raise ValueError('参赛信息不能含有 []{}; 定界符')
        try:
            text.encode('gb2312')
        except UnicodeEncodeError as exc:
            raise ValueError('参赛信息含 GB2312 不支持的字符，请移除表情或生僻字符') from exc
        result[name] = text
    if not result['redTeam'] or not result['blueTeam']:
        raise ValueError('先后手队名不能为空')
    if result['result'] not in ('', *RESULTS):
        raise ValueError('比赛结果只能是先手胜或后手胜')
    return result


def validate_moves(moves, meta):
    if not isinstance(moves, list) or len(moves) > 121:
        raise ValueError('11×11 棋谱最多 121 手')
    game = HexGame(11)
    board, player, winner = game.getInitBoard(), 1, None
    normalized = []
    for i, item in enumerate(moves):
        if not isinstance(item, dict):
            raise ValueError(f'第 {i+1} 手格式无效')
        action, mover = item.get('action'), item.get('player')
        if winner:
            raise ValueError(f'第 {i+1} 手无效：之前已连通获胜')
        if isinstance(mover, bool) or mover != player:
            raise ValueError(f'第 {i+1} 手颜色错误：必须红先蓝后交替落子')
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 121:
            raise ValueError(f'第 {i+1} 手坐标超出 A～K、1～11')
        if board.flat[action]:
            raise ValueError(f'第 {i+1} 手重复占用 {coordinate(action)}')
        move = dict(move=i+1, player=player, action=action, row=action//11, col=action%11,
                    actor='record', source='record')
        mark = item.get('mark')
        if mark is not None:
            if isinstance(mark, bool) or not isinstance(mark, int) or mark not in LABELS:
                raise ValueError(f'第 {i+1} 手 MARK 必须是 -2、-1、0、1、2')
            move['mark'] = mark
        comment = item.get('comment', '')
        if not isinstance(comment, str) or len(comment) > 4000:
            raise ValueError('单手详细评语最多 4000 字')
        if comment:
            move['comment'] = comment
        # 详细分析仅作为 JSON 数据处理，绝不执行导入内容。
        if isinstance(item.get('review'), dict):
            move['review'] = item['review']
        normalized.append(move)
        board, player = game.getNextState(board, player, action)
        ended = game.getGameEnded(board, player)
        winner = int(ended * player) if ended else None
    actual = '先手胜' if winner == 1 else '后手胜' if winner == -1 else ''
    if actual and meta['result'] and actual != meta['result']:
        raise ValueError('棋谱声明的胜方与棋盘实际连通结果矛盾')
    warnings = []
    if meta['result'] and not actual:
        warnings.append('棋谱声明了胜方，但所录局面尚未连通；可能为认输、裁定或节选，软件无法据此验证胜负。')
    return normalized, winner, warnings


def parse_txt(data):
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError('棋谱文件超过 256 KB')
    try:
        text = data.decode('gb2312').strip()
    except UnicodeDecodeError as exc:
        raise ValueError('TXT 棋谱必须使用 GB2312 编码，不能使用 UTF-8 或 UTF-16') from exc
    if not text.startswith('{') or not text.endswith('}'):
        raise ValueError('棋谱必须用英文 { 和 } 包围')
    parts = text[1:-1].split(';')
    header = parts.pop(0).strip()
    fields = re.findall(r'\[([^\[\]{};]*)\]', header)
    if len(fields) != 6 or re.sub(r'\[[^\[\]{};]*\]', '', header).strip():
        raise ValueError('棋谱头必须包含 [HEX][先手队 R][后手队 B][结果][时间地点][赛事]')
    fields = [' '.join(s.split()) for s in fields]
    if fields[0] != 'HEX' or not fields[1].endswith(' R') or not fields[2].endswith(' B'):
        raise ValueError('必须是 HEX 棋谱，第一队以 R 标记红方先手，第二队以 B 标记蓝方后手')
    if fields[3] not in RESULTS:
        raise ValueError('比赛 TXT 的结果字段必须为先手胜或后手胜')
    meta = metadata_checked(dict(redTeam=fields[1][:-2], blueTeam=fields[2][:-2],
                                 result=fields[3], timePlace=fields[4], event=fields[5]))
    moves = []
    for i, part in enumerate(parts):
        match = re.fullmatch(r'\s*([RB])\(\s*([A-K])\s*,\s*([0-9]+)\s*\)\s*(?:MARK\[([+-]?[0-9]+)\])?\s*', part)
        if not match:
            raise ValueError(f'第 {i+1} 手格式错误，示例：R(E,7)MARK[1]')
        color, col, row, mark = match.groups()
        row = int(row)
        if not 1 <= row <= 11:
            raise ValueError(f'第 {i+1} 手纵坐标必须为 1～11')
        move = dict(player=1 if color == 'R' else -1, action=(11-row)*11+ord(col)-65)
        if mark is not None:
            move['mark'] = int(mark)
        moves.append(move)
    normalized, winner, warnings = validate_moves(moves, meta)
    return dict(metadata=meta, moves=normalized, winner=winner, warnings=warnings)


def parse_json_record(data):
    if len(data) > MAX_JSON_RECORD_BYTES:
        raise ValueError('详细 JSON 棋谱超过 8 MB')
    try:
        record = json.loads(data.decode('utf-8-sig'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError('详细棋谱 JSON 必须是有效 UTF-8 文本') from exc
    if not isinstance(record, dict) or record.get('boardSize') != 11:
        raise ValueError('仅支持 11×11 Hex JSON 棋谱')
    meta = metadata_checked(record.get('recordMetadata', {}))
    moves, winner, warnings = validate_moves(record.get('recordMoves', record.get('moves')), meta)
    if record.get('format') != 'hexzero-local-match-v3':
        warnings.append('旧版 JSON 按先后手映射为红先蓝后，内部棋盘坐标保持原样。')
    return dict(metadata=meta, moves=moves, winner=winner, warnings=warnings)


def export_txt(moves, metadata):
    meta = metadata_checked(metadata)
    moves, winner, _ = validate_moves(moves, meta)
    if winner:
        meta['result'] = RESULTS[0 if winner == 1 else 1]
    if not meta['result']:
        raise ValueError('比赛 TXT 必须填写先手胜或后手胜；未结束且未裁定的棋局请先保存详细 JSON')
    head = f"[HEX][{meta['redTeam']} R][{meta['blueTeam']} B][{meta['result']}][{meta['timePlace']}][{meta['event']}]"
    tokens = [head]
    for move in moves:
        token = f"{'R' if move['player'] == 1 else 'B'}({coordinate(move['action'])})"
        if move.get('mark') is not None:
            token += f"MARK[{move['mark']}]"
        tokens.append(token)
    data = ('{'+';'.join(tokens)+'}').encode('gb2312')
    filename = f"HEX-{meta['redTeam']} R vs {meta['blueTeam']} B-{meta['result']}-{meta['timePlace']}-{meta['event']}.txt"
    filename = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', filename)
    if len(filename) > 200:
        raise ValueError('文件名过长，请缩短队名、赛事或时间地点')
    return filename, data
