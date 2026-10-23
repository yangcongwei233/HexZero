"""Hex 对弈业务核心：桌面窗口与旧网页入口共用，不依赖 HTTP 或 Tk。

状态约定：+1 为先手红方（连接上下），-1 为后手蓝方（连接左右）。
board[row, col] 的 row 从上往下增加；比赛坐标为字母 col 与 n-row。

HexApp 管理网络、模型目录与会话；HexSession 管理单局、搜索树、棋钟和棋谱。
界面只能通过这些方法修改棋局，不能直接改 board，否则两个引擎的树会失步。
方法调用需要由调用方串行化：网页使用 app.lock，桌面使用单工作线程。
revision 用于拒绝基于旧棋盘的命令；分析使用独立搜索树，不增加实战搜索预算。
"""
import base64
import copy
from collections import deque
import json
import logging
from pathlib import Path
import re
from threading import Lock
import time
import uuid

import numpy as np

from hex.HexGame import HexGame
from hex.HexLogic import HEX_DIRECTIONS
from hex.local_players import AzaleaReference, TrainedPlayer
from hex.local_analysis import tree_candidates
from hex.match_clock import MatchClock
from hex.record_format import (parse_txt, parse_json_record, export_txt, metadata_checked,
                               coordinate, mark_from_value, LABELS, MAX_RECORD_BYTES,
                               MAX_JSON_RECORD_BYTES)

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'{name}必须是 {low}～{high} 的整数')
    return value


def latest_checkpoint(folder):
    candidates = []
    for path in Path(folder).glob('checkpoint_*.pth.tar'):
        match = re.fullmatch(r'checkpoint_(\d+)\.pth\.tar', path.name)
        if match:
            candidates.append((int(match[1]), path))
    if candidates:
        return max(candidates)[1]
    for name in ('trained.pth.tar', 'best.pth.tar', 'latest.pth.tar'):
        path = Path(folder) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'找不到训练模型：{folder}；请用 --checkpoint 指定你的模型')


def winning_path(board, winner):
    if winner is None:
        return []
    n = len(board)
    starts = [(0, c) for c in range(n)] if winner == 1 else [(r, 0) for r in range(n)]
    parent = {cell: None for cell in starts if board[cell] == winner}
    queue = deque(parent)
    while queue:
        cell = queue.popleft()
        row, col = cell
        if (row if winner == 1 else col) == n - 1:
            path = []
            while cell is not None:
                path.append(list(cell))
                cell = parent[cell]
            return path[::-1]
        for dr, dc in HEX_DIRECTIONS:
            other = row + dr, col + dc
            if (0 <= other[0] < n and 0 <= other[1] < n
                    and other not in parent and board[other] == winner):
                parent[other] = cell
                queue.append(other)
    return []


class HexSession:
    """单局状态机：普通对局、手动摆开局、只读打谱三种状态互斥。

    每个引擎具有独立随机数状态与搜索树；mine 只决定用户/参考模型执哪方，
    不改变红先蓝后规则。复盘 seek 不走真实引擎；续弈会重建引擎并重放前缀。
    """
    def __init__(self, app, mode, mine, simulations, seed, manual_opening=False, move_time_ms=0, clock_minutes=0):
        self.app, self.game = app, app.game
        self.move_time_ms = app.validate_move_time(move_time_ms)
        self.clock = MatchClock(clock_minutes)
        self.winner_reason = None
        self.id = uuid.uuid4().hex
        self.mode, self.mine, self.simulations, self.seed = mode, mine, simulations, seed
        self.board = self.game.getInitBoard()
        self.player, self.winner = 1, None
        self.moves = []
        self.revision = 0
        self.saved_path = None
        self.opening_setup = manual_opening
        self.opening_length = 0
        self.analysis = None
        self.created_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        teams = {'human': '人类', 'azalea': 'Azalea', 'trained': 'HexZero'}
        self.record_metadata = dict(redTeam=teams.get(self.kind(1), '先手队'),
                                    blueTeam=teams.get(self.kind(-1), '后手队'), result='',
                                    timePlace=time.strftime('%Y.%m.%d %H:%M') + ' 本地', event='本地对弈')
        self.replay_mode = False
        self.record_moves = None
        self.record_warnings = []
        self.engines = self.make_engines()

    def make_engines(self):
        """HexZero 使用 seed，Azalea 使用 seed+1，避免共享随机数流。

        此约定沿用旧版棋谱。即使温度为 0，访问次数并列仍可能消耗随机数；
        分析和实战各自持有引擎，不能通过点击分析提前改变实战的随机选择。
        """
        # 人人对弈不构造网络或搜索树，公开源码在没有权重时也能完整打谱和下棋。
        if self.mode == 'pvp':
            return {}
        engines = {-self.mine: TrainedPlayer(self.game, self.app.net, self.simulations, self.seed,
                                             move_time_ms=self.move_time_ms)}
        if self.mode == 'azalea':
            engines[self.mine] = self.app.reference.player(self.simulations, (self.seed + 1) % (2 ** 32))
        return engines

    def kind(self, player):
        if self.mode == 'pvp':
            return 'human'
        return self.mode if player == self.mine else 'trained'

    def state(self):
        """生成界面所需状态，并检查真实棋钟是否到期。

        此函数可能因超时更新 winner/revision，不能在搜索工作线程之外并发调用。
        网页将其序列化为 JSON，桌面服务在工作线程深拷贝后交给主线程。
        """
        self.refresh_clock()
        can_undo = bool(self.moves) if self.opening_setup or self.mode == 'azalea' else any(
            m['actor'] == 'human' or m.get('source') == 'manual_opening' for m in self.moves)
        return dict(id=self.id, boardSize=self.game.n, board=self.board.tolist(),
                    currentPlayer=self.player, currentKind=self.kind(self.player),
                    myPlayer=self.mine, mode=self.mode, winner=self.winner,
                    winnerKind=self.kind(self.winner) if self.winner else None,
                    winningPath=winning_path(self.board, self.winner),
                    moves=list(self.moves), revision=self.revision, mctsSims=self.simulations,
                    moveTimeMs=self.move_time_ms, clock=self.clock.state(), winnerReason=self.winner_reason,
                    seed=self.seed, canUndo=can_undo and not self.replay_mode, savedPath=self.saved_path,
                    openingSetup=self.opening_setup, openingLength=self.opening_length,
                    analysis=self.analysis, replayMode=self.replay_mode,
                    replayCursor=len(self.moves), replayTotal=len(self.record_moves) if self.replay_mode else len(self.moves),
                    replayMoves=[{k: m[k] for k in ('move', 'player', 'action', 'row', 'col', 'actor', 'source', 'mark') if k in m}
                                 for m in self.record_moves] if self.replay_mode else [],
                    recordMetadata=self.record_metadata.copy(), recordWarnings=list(self.record_warnings),
                    notation=dict(origin='bottom-left', first='R', second='B', redConnects='top-bottom'))

    def check_revision(self, revision):
        self.refresh_clock()
        if revision != self.revision:
            raise ValueError('棋局已更新，请刷新后重试')
        if self.winner:
            raise ValueError('本局已结束，请新开一局或悔棋')

    def refresh_clock(self):
        if not self.winner and not self.replay_mode and not self.opening_setup:
            expired = self.clock.expired()
            if expired:
                self.winner, self.winner_reason = -expired, 'timeout'
                self.clock.pause()
                self.revision += 1
                self.record_metadata['result'] = '先手胜' if self.winner == 1 else '后手胜'
                self.save()

    def control_clock(self, running, revision):
        self.check_revision(revision)
        if type(running) is not bool or not self.clock.minutes or self.opening_setup or self.replay_mode:
            raise ValueError('请在启用计时的实战局面中操作棋钟')
        if running:
            self.clock.start(self.player)
        else:
            self.clock.pause()
        self.revision += 1

    def begin_turn(self):
        if self.clock.minutes:
            if not self.clock.started:
                self.clock.start(self.player)
            elif self.clock.active is None:
                raise ValueError('棋钟已暂停，请先继续计时')

    def apply(self, action, info=None):
        """唯一的真实落子入口，同时维护规则、双方搜索树、棋钟和自动存谱。

        action 是原始棋盘 row*n+col，而非网络规范化坐标；每个玩家适配器
        负责自己的坐标转换。树推进也计入比赛耗时，超时则不能提交该落子。
        """
        player = self.player
        next_board, next_player = self.game.getNextState(self.board, player, int(action))
        for engine in self.engines.values():
            engine.observe(self.board, player, int(action))
        # Advancing/compacting a large tree can also take time. Do not commit a
        # move that exhausted the clock during this work.
        if self.clock.active is not None:
            self.refresh_clock()
            if self.winner:
                return
        self.board, self.player = next_board, next_player
        row, col = divmod(int(action), self.game.n)
        self.moves.append(dict(move=len(self.moves) + 1, actor=self.kind(player), player=player,
                               action=int(action), row=row, col=col, **(info or {})))
        if info and info.get('source') == 'manual_opening':
            self.opening_length = len(self.moves)
        self.revision += 1
        self.analysis = None
        result = self.game.getGameEnded(self.board, self.player)
        self.winner = self.player * result if result else None
        if self.winner:
            self.winner_reason = 'connection'
            self.clock.pause()
        elif self.clock.active is not None:
            self.clock.start(self.player)
        if self.clock.minutes and 'clockAfter' not in self.moves[-1]:
            self.moves[-1]['clockAfter'] = self.clock.state()
        self.saved_path = None
        if self.winner:
            self.save()

    def human_move(self, row, col, revision):
        self.check_revision(revision)
        if self.replay_mode:
            raise ValueError('打谱查看模式不能直接落子，请先选择“从此局面续弈”')
        if not self.opening_setup and self.kind(self.player) != 'human':
            raise ValueError('当前轮到模型落子')
        integer(row, '行', 0, self.game.n - 1)
        integer(col, '列', 0, self.game.n - 1)
        if self.board[row, col]:
            raise ValueError('该位置已有棋子')
        if not self.opening_setup:
            self.begin_turn()
        self.apply(row * self.game.n + col,
                   {'source': 'manual_opening'} if self.opening_setup else None)

    def finish_opening(self, revision):
        self.check_revision(revision)
        if not self.opening_setup:
            raise ValueError('当前不在手动开局阶段')
        self.opening_setup = False
        self.revision += 1
        self.analysis = None
        self.saved_path = None

    def analyze(self, revision, kind='current'):
        """用独立树复盘到当前局面，搜索后提取最多三条已探索主变化。

        不给实战树追加访问次数，不修改实战 RNG；Q 以当前执棋方为正方向。
        相同局面与分析引擎的结果可缓存；任何真实落子、悔棋、跳谱都会失效。
        """
        self.check_revision(revision)
        if kind == 'current':
            kind = self.kind(self.player)
            if kind == 'human':
                kind = 'trained'
        if kind not in ('trained', 'azalea'):
            raise ValueError('分析模型只能选择 trained、azalea 或 current')
        if kind == 'azalea' and self.app.reference is None:
            raise ValueError(self.app.reference_error)
        if kind == 'trained' and self.app.net is None:
            raise ValueError('尚未加载模型；可先人人对弈，放入 ONNX 模型后再使用分析')
        if self.analysis and self.analysis['revision'] == revision and self.analysis['engine'] == kind:
            return
        started = time.perf_counter()
        # Separate RNG and search tree: inspecting a position must not give the
        # playing engine extra simulations or alter its next tie-breaking draw.
        engine = (TrainedPlayer(self.game, self.app.net, self.simulations, self.seed)
                  if kind == 'trained' else
                  self.app.reference.player(self.simulations, (self.seed + 1) % (2 ** 32)))
        board, player = self.game.getInitBoard(), 1
        for move in self.moves:
            engine.observe(board, player, move['action'])
            board, player = self.game.getNextState(board, player, move['action'])
        # Analysis observes the same per-move cap, but never borrows live-tree visits.
        limit = self.clock.budget(self.player, int((self.board == 0).sum()), self.move_time_ms / 1000)
        _, info = engine.choose(self.board, self.player, time_limit=limit)
        candidates = tree_candidates(engine, self.board, self.player, self.game, kind)
        self.analysis = dict(revision=self.revision, engine=kind, player=self.player,
                             simulations=self.simulations, maxDepth=6, ranking='visits_then_value_then_prior',
                             seconds=time.perf_counter() - started, candidates=candidates,
                             rootValue=info['value'], completedSimulations=info.get('searchSimulations'),
                             stopReason=info.get('searchStopReason'))
        self.saved_path = None
        self.refresh_clock()

    def step(self, revision):
        self.check_revision(revision)
        if self.replay_mode:
            raise ValueError('打谱查看模式不能自动落子')
        if self.opening_setup:
            raise ValueError('请先完成手动开局，再让模型落子')
        if self.kind(self.player) == 'human':
            raise ValueError('请在棋盘上落子')
        self.begin_turn()
        limit = self.clock.budget(self.player, int((self.board == 0).sum()), self.move_time_ms / 1000)
        action, info = self.engines[self.player].choose(self.board, self.player, time_limit=limit)
        self.refresh_clock()
        if self.winner:
            return
        info['searchTimeBudgetMs'] = round(limit * 1000)
        self.apply(action, info)

    def undo(self, revision):
        """重建引擎再重放保留棋谱，避免沿用已经被裁剪或属于未来的树节点。

        人机对弈撤销人的上一手及其后的模型回应；人人对弈/手动开局撤销一手。
        如果退回手动前缀，将重新开放摆棋，不立即自动补回刚撤销的模型落子。
        """
        if revision != self.revision:
            raise ValueError('棋局已更新，请刷新后重试')
        if not self.state()['canUndo']:
            raise ValueError('没有可以撤销的落子')
        keep = len(self.moves) - 1
        if self.mode == 'human' and not self.opening_setup:
            human_moves = [i for i, m in enumerate(self.moves)
                           if m['actor'] == 'human' and m.get('source') != 'manual_opening']
            if human_moves:
                keep = human_moves[-1]
        retained = self.moves[:keep]
        clock_snapshot = (retained[-1].get('clockAfter') if retained else None)
        self.clock.pause()
        self.winner_reason = None
        # Returning to the manually placed prefix allows editing either color
        # without an automatic model response immediately replacing the undo.
        self.opening_setup = self.opening_setup or (self.opening_length > 0 and keep <= self.opening_length)
        self.opening_length = 0
        # Search statistics and reference environment must follow the restored board.
        self.engines = self.make_engines()
        self.board, self.player = self.game.getInitBoard(), 1
        self.moves, self.winner, self.saved_path = [], None, None
        self.analysis = None
        for move in retained:
            self.apply(move['action'], {k: move[k] for k in ('seconds', 'value', 'source', 'opening_id', 'mark', 'comment', 'review',
                                                            'searchSimulations', 'searchStopReason', 'moveTimeMs', 'searchTimeBudgetMs', 'clockAfter') if k in move})
        self.revision += 1
        self.clock = MatchClock(self.clock.minutes)
        if clock_snapshot:
            self.clock.restore(clock_snapshot)

    def seek(self, ply, revision):
        """只读跳谱：保存完整 record_moves，显示的 moves 只是前 ply 手。

        跳到中途也不会截断导出的完整棋谱；“从此局面续弈”另建会话，
        从而保留原始棋谱与评语。此时不会自动运行模型，棋钟暂停。
        """
        if revision != self.revision:
            raise ValueError('棋局已更新，请刷新后重试')
        integer(ply, '打谱手数', 0, len(self.record_moves) if self.replay_mode else len(self.moves))
        if not self.replay_mode:
            self.clock.pause()
            self.record_moves = copy.deepcopy(self.moves)
            self.replay_mode = True
        board, player, winner = self.game.getInitBoard(), 1, None
        for move in self.record_moves[:ply]:
            board, player = self.game.getNextState(board, player, move['action'])
            result = self.game.getGameEnded(board, player)
            winner = int(result * player) if result else None
        self.board, self.player, self.winner = board, player, winner
        self.winner_reason = 'connection' if winner else None
        self.moves = copy.deepcopy(self.record_moves[:ply])
        self.opening_setup = False
        self.analysis = None
        self.revision += 1

    def update_metadata(self, metadata, revision):
        if revision != self.revision:
            raise ValueError('棋局已更新，请刷新后重试')
        checked = metadata_checked(metadata)
        # Validate against the whole record, even while displaying an early ply.
        if self.game.n == 11:
            from hex.record_format import validate_moves
            _, winner, warnings = validate_moves(self.record_moves if self.replay_mode else self.moves, checked)
            self.record_warnings = warnings
        self.record_metadata = checked
        self.save()

    def annotate(self, revision, mark=None, comment='', generated=False):
        """评注属于刚下完的那一手，自动评分需把当前方 Q 翻成落子方视角。

        MARK 是局面等级而非“这一手导致的损失”；评价单手质量需要比较落子前
        的候选和落子后估值，不能仅凭这里一个 Q 值认定失误。
        """
        if revision != self.revision or not self.moves:
            raise ValueError('请先选择一手有效棋子，再添加评注')
        if mark is not None:
            integer(mark, 'MARK', -2, 2)
        if not isinstance(comment, str) or len(comment) > 4000:
            raise ValueError('详细评语最多 4000 字')
        if not isinstance(generated, bool):
            raise ValueError('自动评注设置必须是布尔值')
        move = self.moves[-1]
        review = None
        if generated:
            analysis = self.analysis
            if not analysis or analysis['revision'] != self.revision:
                raise ValueError('请先分析当前局面，再生成评注')
            value = float(analysis['rootValue']) * self.player * move['player']
            mark = mark_from_value(value)
            color = '红方' if move['player'] == 1 else '蓝方'
            next_color = '红方' if self.player == 1 else '蓝方'
            engine = 'HexZero 训练模型' if analysis['engine'] == 'trained' else 'Azalea 参考模型'
            lines = [f"第 {len(self.moves)} 手 {'R' if move['player'] == 1 else 'B'}({coordinate(move['action'], self.game.n)})，落子后从{color}视角评为 {mark:+d}（{LABELS[mark]}）。",
                     f'{engine}，{self.simulations} 次搜索设置；{color}视角 Q={value:+.4f}。这是局面优劣，不能单独证明本手好坏，也不是校准胜率。',
                     f'接下来轮到{next_color}。其候选按访问次数排序：']
            for candidate in analysis['candidates']:
                q = candidate['value']
                score = '尚未搜索' if q is None else f"Q={q:+.4f}，{mark_from_value(q):+d}（{LABELS[mark_from_value(q)]}）"
                pv = ' → '.join(f"{'R' if m['player'] == 1 else 'B'}({coordinate(m['action'], self.game.n)})" for m in candidate['pv'])
                lines.append(f"{candidate['rank']}. {coordinate(candidate['action'], self.game.n)}：{next_color}视角 {score}；访问 {candidate['visits']} 次。后续：{pv}。")
            lines.append('分档阈值：Q<−0.6 为大劣；−0.6≤Q<−0.2 为小劣；−0.2≤Q≤0.2 为一般；0.2<Q≤0.6 为小好；Q>0.6 为大好。主变化只显示已搜索分支。')
            comment = '\n'.join(lines)
            review = copy.deepcopy(analysis)
            review.update(markPerspective='last_mover', moverValue=value)
        move.pop('mark', None)
        if mark is not None:
            move['mark'] = mark
        move['comment'] = comment
        if review is not None:
            move['review'] = review
        else:
            move.pop('review', None)
        if self.replay_mode:
            self.record_moves[len(self.moves)-1] = copy.deepcopy(move)
        self.save()

    def competition_record(self):
        if self.game.n != 11:
            raise ValueError('比赛 TXT 格式仅用于 11×11 棋盘')
        return export_txt(self.record_moves if self.replay_mode else self.moves, self.record_metadata)

    def record(self):
        config = self.app.config()
        return dict(format='hexzero-local-match-v3', createdAt=self.created_at,
                    models=config['models'], search=config['search'], inference=config['inference'],
                    recordMoves=copy.deepcopy(self.record_moves if self.replay_mode else self.moves), **self.state())

    def save(self):
        folder = self.app.save_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'hex_{self.id}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record(), ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
        self.saved_path = str(path)


class HexApp:
    """模型与会话的生命周期管理器，允许网页和原生桌面共享完全相同的棋力。

    ONNX 模型保持 FP32；线程数只改变推理运行配置，不改变网络权重。
    换模型先校验输出并保存旧棋局，再建立新会话，失败时恢复旧模型对象。
    """
    def __init__(self, checkpoint, save_dir, device='cpu', simulations=3200,
                 reference_source=None, reference_checkpoint=None, board_size=None,
                 backend='torch', threads=None):
        self.checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
        self.net = None
        if backend not in ('torch', 'onnx'):
            raise ValueError('未知推理后端')
        self.backend = backend
        if self.checkpoint is None:
            # 无模型时只提供规则与棋谱；不导入 ONNX Runtime/PyTorch。
            state = dict(board_size=[board_size or 11] * 2, metadata={})
            self.threads = threads if threads is not None else 4
        elif backend == 'onnx':
            if device != 'cpu':
                raise ValueError('ONNX 本地对弈后端只支持 CPU')
            from hex.onnx_inference import OnnxNetwork
            self.net = OnnxNetwork(self.checkpoint, 'trained', threads)
            state = dict(board_size=[self.net.board_size] * 2,
                         metadata=self.net.checkpoint_metadata.copy())
            self.threads = self.net.threads
        else:
            import torch
            from hex.pytorch.NNet import NNetWrapper
            state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
            self.threads = torch.get_num_threads()
        size = state.get('board_size')
        if size is None:
            if board_size is None:
                raise ValueError('旧模型缺少棋盘尺寸，请指定 --board-size')
            size = [board_size, board_size]
        if len(size) != 2 or size[0] != size[1] or (board_size and size[0] != board_size):
            raise ValueError('模型棋盘尺寸与 --board-size 不匹配')
        self.game = HexGame(int(size[0]))
        if backend == 'torch' and self.checkpoint is not None:
            self.net = NNetWrapper(self.game, config={'cuda': device == 'cuda'})
            # Playing needs neither optimizer nor training RNG state.
            self.net.load_training_state({k: v for k, v in state.items()
                                          if k not in ('optimizer_state_dict', 'rng_state')})
        self.save_dir = Path(save_dir).resolve()
        self.device, self.default_simulations = device, simulations
        self.reference = None
        self.reference_error = None
        if reference_checkpoint and Path(reference_checkpoint).is_file():
            try:
                if backend == 'onnx':
                    from hex.onnx_inference import OnnxReference
                    self.reference = OnnxReference(reference_checkpoint, self.game.n, self.threads)
                else:
                    self.reference = AzaleaReference(reference_source, reference_checkpoint, self.game.n, device)
            except (ImportError, ValueError, KeyError, OSError) as exc:
                self.reference_error = str(exc)
                log.warning('Azalea reference unavailable: %s', exc)
        else:
            self.reference_error = '未找到 azalea 参考权重，请指定 --reference-checkpoint'
        self.metadata = state.get('metadata', {})
        manifest = self.checkpoint.with_suffix(self.checkpoint.suffix + '.json') if self.checkpoint else None
        if backend == 'torch' and manifest is not None and manifest.is_file():
            self.metadata.update(json.loads(manifest.read_text()))
        self.sessions = {}
        self.lock = Lock()
        self.on_threads_changed = None
        self.model_library = None
        self.on_model_changed = None

    def enable_model_library(self, directory, on_change=None):
        if self.backend != 'onnx':
            raise ValueError('模型目录选择需要 ONNX 推理后端')
        from hex.model_library import ModelLibrary
        self.model_library = ModelLibrary(directory, self.game.n)
        if self.checkpoint is not None:
            self.model_library.model_id(self.checkpoint)
        self.on_model_changed = on_change

    def model_catalog(self, refresh=False):
        if self.model_library is None:
            return None
        result = self.model_library.refresh() if refresh else self.model_library.snapshot()
        result.update(selectedId=self.model_library.model_id(self.checkpoint) if self.checkpoint else None,
                      activeSha256=self.net.manifest['onnx_sha256'] if self.net else None)
        return result

    def select_model(self, ident, mode='human', mine=1, simulations=None, seed=0, move_time_ms=0, clock_minutes=0):
        if self.model_library is None:
            raise ValueError('未配置模型目录')
        move_time_ms = self.validate_move_time(move_time_ms)
        simulations = self.validate_new_session(mode, mine, simulations, seed, False, require_model=False)
        from hex.onnx_inference import OnnxNetwork, validate_trained_network
        checkpoint = self.model_library.resolve(ident)
        try:
            net = OnnxNetwork(checkpoint, 'trained', self.threads)
            validate_trained_network(net)
        except Exception as exc:
            raise ValueError(f'模型试运行失败，未切换：{exc}') from exc
        # Save with the original model metadata before replacing any app state.
        for session in self.sessions.values():
            session.save()
        previous = self.checkpoint, self.net, self.metadata
        self.checkpoint, self.net, self.metadata = checkpoint, net, net.checkpoint_metadata.copy()
        try:
            session = HexSession(self, mode, mine, simulations, seed, move_time_ms=move_time_ms, clock_minutes=clock_minutes)
            if self.on_model_changed:
                self.on_model_changed(ident)
        except Exception:
            self.checkpoint, self.net, self.metadata = previous
            raise
        self.sessions = {session.id: session}
        return dict(config=self.config(), session=session.state())

    def set_threads(self, threads):
        integer(threads, 'CPU 推理线程数', 4, 16)
        if self.backend != 'onnx':
            raise ValueError('此设置仅适用于优化后的 ONNX 推理版')
        if self.net is None:
            self.threads = threads
            return self.config()
        if threads != self.threads:
            from hex.onnx_inference import OnnxNetwork
            # Construct both sessions before committing. Live players share these
            # network objects, so their existing trees and game state stay valid.
            replacements = [(self.net, OnnxNetwork(self.checkpoint, 'trained', threads))]
            if self.reference:
                replacements.append((self.reference.prototype.net,
                                     OnnxNetwork(self.reference.checkpoint, 'azalea', threads)))
            if any(old.manifest['onnx_sha256'] != new.manifest['onnx_sha256'] for old, new in replacements):
                raise ValueError('当前模型文件已被替换，请重新选择模型并新开局')
            if self.on_threads_changed:
                self.on_threads_changed(threads)
            for current, updated in replacements:
                current.session, current.threads = updated.session, threads
            self.threads = threads
        return self.config()

    def config(self):
        return dict(boardSize=self.game.n, device=self.device, defaultSimulations=self.default_simulations,
                    inference=dict(backend=self.backend, threads=self.threads, precision='FP32',
                                   configurable=self.backend == 'onnx', minThreads=4, maxThreads=16),
                    modelAvailable=self.net is not None,
                    referenceAvailable=self.reference is not None, referenceError=self.reference_error,
                    modelLibrary=self.model_catalog(),
                    models=dict(trained=dict(name=self.checkpoint.name, path=str(self.checkpoint),
                                             metadata=self.metadata) if self.checkpoint else None,
                                azalea=dict(name=self.reference.checkpoint.name, path=str(self.reference.checkpoint))
                                if self.reference else None),
                    search=dict(trained='HexZero batched MCTS, batch 32, cpuct 0.5',
                                azalea='Original Azalea MCTS and checkpoint search settings',
                                noise=False, maxSimulations=100000, maxMoveTimeMs=120000))

    def validate_new_session(self, mode, mine, simulations, seed, manual_opening, require_model=True):
        if mode not in ('human', 'pvp', 'azalea'):
            raise ValueError('对局模式必须为 human、pvp 或 azalea')
        if require_model and mode != 'pvp' and self.net is None:
            raise ValueError('尚未加载模型，请先人人对弈或在设置中加载 ONNX 模型')
        integer(mine, '先后手', -1, 1)
        if mine not in (-1, 1):
            raise ValueError('先后手必须是 1 或 -1')
        simulations = self.default_simulations if simulations is None else simulations
        integer(simulations, '搜索次数', 1, 100000)
        integer(seed, '随机种子', 0, 2 ** 32 - 1)
        if not isinstance(manual_opening, bool):
            raise ValueError('手动开局设置必须是布尔值')
        if mode == 'azalea' and self.reference is None:
            raise ValueError(self.reference_error)
        return simulations

    @staticmethod
    def validate_move_time(value):
        integer(value, '单步用时', 0, 120000)
        if 0 < value < 100:
            raise ValueError('单步用时至少为 100 毫秒，或用 0 表示固定搜索次数')
        return value

    def new_session(self, mode='human', mine=1, simulations=None, seed=0, manual_opening=False, move_time_ms=0, clock_minutes=0):
        simulations = self.validate_new_session(mode, mine, simulations, seed, manual_opening)
        session = HexSession(self, mode, mine, simulations, seed, manual_opening, move_time_ms, clock_minutes)
        for previous in self.sessions.values():
            previous.refresh_clock()
            previous.clock.pause()
        # Bound retained sessions; search trees grow with the explored positions.
        while len(self.sessions) >= 2:
            self.sessions.pop(next(iter(self.sessions)))
        self.sessions[session.id] = session
        return session.state()

    def import_record(self, encoded, kind, mode='human', mine=1, simulations=None, seed=0, move_time_ms=0, clock_minutes=0):
        if self.game.n != 11:
            raise ValueError('比赛棋谱仅支持 11×11 棋盘')
        if kind not in ('txt', 'json'):
            raise ValueError('仅支持 TXT 比赛棋谱或 JSON 详细棋谱')
        byte_limit = MAX_RECORD_BYTES if kind == 'txt' else MAX_JSON_RECORD_BYTES
        if not isinstance(encoded, str) or len(encoded) > 4 * ((byte_limit + 2) // 3):
            raise ValueError('棋谱文件过大或格式无效')
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('棋谱文件传输失败') from exc
        parsed = parse_txt(data) if kind == 'txt' else parse_json_record(data)
        state = self.new_session(mode, mine, simulations, seed, move_time_ms=move_time_ms, clock_minutes=clock_minutes)
        session = self.sessions[state['id']]
        session.record_metadata = parsed['metadata']
        session.record_moves = parsed['moves']
        session.record_warnings = parsed['warnings']
        session.replay_mode = True
        session.seek(0, session.revision)
        return session.state()

    def continue_record(self, session, revision):
        if not session.replay_mode or session.winner or revision != session.revision:
            raise ValueError('请选择尚未结束的打谱局面再续弈')
        session.save()
        state = self.new_session(session.mode, session.mine, session.simulations, session.seed, True, session.move_time_ms, session.clock.minutes)
        branch = self.sessions[state['id']]
        branch.record_metadata = dict(session.record_metadata, result='')
        for move in session.moves:
            branch.apply(move['action'], {k: copy.deepcopy(v) for k, v in move.items()
                                         if k in ('mark', 'comment', 'review') } | {'source': 'manual_opening'})
        return branch.state()
