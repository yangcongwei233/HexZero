import json
import copy
import logging
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from pickle import Pickler, Unpickler
from random import shuffle

from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from hex.value_diagnostics import log_example_value_stats

log = logging.getLogger(__name__)


def arg_value(args, name, default):
    """读取训练配置项，不存在时使用默认值，兼容旧断点缺少的新字段。"""
    return getattr(args, name, default)


def atomic_pickle(path, payload):
    """先将经验数据写入同目录临时文件，再原子替换目标；写入失败时清理临时文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.examples-', delete=False) as f:
        temporary = f.name
        try:
            Pickler(f).dump(payload)
        except BaseException:
            os.unlink(temporary)
            raise
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_alias(source, target):
    """用同目录硬链接发布别名，避免重写大文件。"""
    fd, temporary = tempfile.mkstemp(dir=Path(target).parent, prefix='.alias-')
    os.close(fd)
    os.unlink(temporary)
    try:
        os.link(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Coach:
    """串联自博弈、经验回放、网络更新和平衡先后手评估。"""

    def __init__(self, game, nnet, args):
        """绑定规则、网络和配置，建立旧模型副本及搜索器。按配置加载并校验教师集，初始化可选的固定容量经验池。"""
        self.game, self.nnet, self.args = game, nnet, args
        self.pnet = copy.deepcopy(nnet)
        self.mcts = self._new_mcts(nnet)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.replayBuffer = None
        self.writer = None
        self._episode_index = 0
        self.teacher = None
        teacher_fraction = arg_value(args, 'teacherFraction', 0.)
        if not 0 <= teacher_fraction < 1:
            raise ValueError('Teacher fraction must be in [0, 1)')
        if teacher_fraction:
            from hex.teacher_data import TeacherDataset
            if not arg_value(args, 'teacherData', None):
                raise ValueError('A positive teacher fraction requires --teacher-data')
            self.teacher = TeacherDataset(args.teacherData, game.n, arg_value(args, 'teacherDataSha256', None))
            args.teacherData = self.teacher.path
            args.teacherDataSha256 = self.teacher.sha256
            log.info('Teacher dataset: %s samples=%s fraction=%.3f sha256=%s',
                     self.teacher.path, len(self.teacher), teacher_fraction, self.teacher.sha256)
        if arg_value(args, 'replayBufferSize', 0):
            from hex.replay_buffer import ReplayBuffer
            self.replayBuffer = ReplayBuffer(args.replayBufferSize)

    def _new_mcts(self, nnet):
        """按 mctsBackend 选择经典逐次搜索或 Azalea 风格批量搜索，绑定指定网络。"""
        if arg_value(self.args, 'mctsBackend', 'classic') == 'azalea':
            from hex.azalea_mcts import AzaleaMCTS
            return AzaleaMCTS(self.game, nnet, self.args)
        return MCTS(self.game, nnet, self.args)

    def _arena_player(self, nnet):
        """将指定网络包装为评估策略，每局独立维护搜索状态并处理先后手坐标。"""
        if arg_value(self.args, 'mctsBackend', 'classic') == 'azalea':
            from hex.azalea_mcts import AzaleaStylePolicy
            return AzaleaStylePolicy(self.game, nnet, self.args)
        from hex.policies import ClassicPolicy
        return ClassicPolicy(self.game, nnet, self.args)

    def executeEpisode(self):
        """用当前网络完成一局自博弈，生成带终局价值的样本；按配置保存棋谱并推进对局种子编号。"""
        from hex.parallel_self_play import _execute_episode, episode_seed, save_game_record
        seed = episode_seed(arg_value(self.args, 'seed', 0), arg_value(self.args, 'iteration', 0), self._episode_index)
        examples, record = _execute_episode(
            self.game, self.nnet, self.args, seed, game_id=self._episode_index,
            checkpoint_file='in_memory',
        )
        self._episode_index += 1
        save_game_record(record, self.args, self.args.checkpoint)
        return examples

    def _scalar(self, name, value, step):
        """在 TensorBoard 已启用时写入一个标量；未启用时直接返回，不创建日志目录。"""
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def learn(self):
        """创建可选的 TensorBoard 写入器并执行训练循环；无论正常结束或异常，都释放日志资源。"""
        if arg_value(self.args, 'tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError('TensorBoard is enabled; install requirements.txt first') from exc
            log_dir = arg_value(self.args, 'tensorboardDir', None) or os.path.join(self.args.checkpoint, 'tensorboard')
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
            self.writer.add_text('config', json.dumps(dict(self.args), ensure_ascii=False, indent=2))
            self.nnet.writer = self.writer
            log.info('TensorBoard log directory: %s', log_dir)
        try:
            self._learn()
        finally:
            if self.writer is not None:
                self.writer.close()
                self.nnet.writer = None
                self.writer = None

    def _learn(self):
        """逐轮完成自博弈、经验入池、网络更新和新旧模型评估。
        候选胜率不足时恢复旧权重及优化器，日志步数保持递增；按保存间隔及结束轮次写入完整快照。"""
        start = arg_value(self.args, 'startIter', 1)
        end = start + self.args.numIters
        interval = arg_value(self.args, 'checkpointInterval', 10)
        if interval < 1 or arg_value(self.args, 'keepCheckpoints', 3) < 1:
            raise ValueError('Checkpoint interval and retention must be positive')
        if self.replayBuffer is not None and not len(self.replayBuffer):
            for history in self.trainExamplesHistory:
                self.replayBuffer.extend(history)
        skip = self.skipFirstSelfPlay
        self.skipFirstSelfPlay = False
        for iteration in range(start, end):
            iteration_start = time.perf_counter()
            self.args.iteration = iteration
            self._episode_index = 0
            log.info('Starting Iter #%s ...', iteration)
            selfplay_start = time.perf_counter()
            if not (skip and iteration == start):
                if arg_value(self.args, 'parallelSelfPlay', False):
                    from hex.parallel_self_play import run_parallel_self_play
                    collected = run_parallel_self_play(self.game, self.nnet, self.args)
                else:
                    collected = deque(maxlen=self.args.maxlenOfQueue)
                    for _ in tqdm(range(self.args.numEps), desc='Self Play'):
                        collected.extend(self.executeEpisode())
                if self.replayBuffer is not None:
                    self.replayBuffer.extend(collected)
                    self.trainExamplesHistory = self.replayBuffer.to_history()
                else:
                    self.trainExamplesHistory.append(collected)
                    self.trainExamplesHistory = self.trainExamplesHistory[-self.args.numItersForTrainExamplesHistory:]
                elapsed = time.perf_counter() - selfplay_start
                self._scalar('selfplay/seconds', elapsed, iteration)
                self._scalar('selfplay/games_per_second', self.args.numEps / max(elapsed, 1e-9), iteration)
                self._scalar('selfplay/samples_collected', len(collected), iteration)
                if arg_value(self.args, 'debugValueDiagnostics', False):
                    log_example_value_stats(f'Iter #{iteration} self-play', collected)
            examples = [example for history in self.trainExamplesHistory for example in history]
            if not examples:
                raise ValueError('No training examples; collect self-play or load a valid replay snapshot')
            shuffle(examples)
            self._scalar('replay/samples', len(examples), iteration)
            self._scalar('replay/mean_value_target', sum(float(e[2]) for e in examples) / len(examples), iteration)

            # 回滚状态留在内存，避免每轮写入训练断点。
            previous = self.nnet.get_training_state()
            self.pnet.load_training_state(previous)
            train_start = time.perf_counter()
            if self.teacher is None:
                metrics = self.nnet.train(examples)
            else:
                metrics = self.nnet.train(examples, teacher=self.teacher, teacher_fraction=self.args.teacherFraction)
                self._scalar('teacher/dataset_samples', len(self.teacher), iteration)
                self._scalar('teacher/target_fraction', self.args.teacherFraction, iteration)
            self._scalar('train/seconds_per_iteration', time.perf_counter() - train_start, iteration)
            for name, value in (metrics or {}).items():
                self._scalar('iteration/' + name, value, iteration)
            arena_start = time.perf_counter()
            arena_workers = arg_value(self.args, 'arenaWorkers', 1)
            if arena_workers > 1:
                from hex.parallel_arena import run_parallel_arena
                pwins, nwins, draws = run_parallel_arena(self.game, self.pnet, self.nnet, self.args)
            else:
                arena = Arena(self._arena_player(self.pnet), self._arena_player(self.nnet), self.game)
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            arena_seconds = time.perf_counter() - arena_start
            winrate = nwins / (pwins + nwins) if pwins + nwins else 0.0
            accepted = bool(pwins + nwins and winrate >= self.args.updateThreshold)
            log.info('NEW/PREV WINS: %s / %s; DRAWS: %s; accepted=%s', nwins, pwins, draws, accepted)
            for name, value in {'new_wins': nwins, 'previous_wins': pwins, 'draws': draws,
                                'new_winrate': winrate, 'accepted': int(accepted),
                                'seconds': arena_seconds,
                                'games_per_second': self.args.arenaCompare / max(arena_seconds, 1e-9),
                                'workers': min(arena_workers, self.args.arenaCompare)}.items():
                self._scalar('arena/' + name, value, iteration)
            if not accepted:
                attempted_steps = self.nnet.train_step
                self.nnet.load_training_state(previous)
                self.nnet.train_step = attempted_steps  # 更新被拒绝时，日志步数仍保持递增。
            self.nnet.checkpoint_metadata.update(iteration=iteration, seed=arg_value(self.args, 'seed', 0), run_config=dict(self.args))
            if iteration % interval == 0 or iteration == end - 1:
                self._save_snapshot(iteration)
            self._scalar('iteration/seconds', time.perf_counter() - iteration_start, iteration)
            if self.writer is not None:
                self.writer.flush()

    def getCheckpointFile(self, iteration):
        """根据轮次生成模型快照文件名，不执行磁盘写入。"""
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration, filename=None):
        """将经验历史和棋盘、坐标、增强版本一起原子保存，标记快照位于完整轮次结束之后。"""
        filename = filename or self.getCheckpointFile(iteration) + '.examples'
        atomic_pickle(Path(self.args.checkpoint) / filename, {
            'format_version': 2, 'augmentation_version': 2, 'canonical_version': 1,
            'board_size': self.game.n, 'iteration': iteration, 'phase': 'post_iteration',
            'history': self.trainExamplesHistory,
        })

    def loadTrainExamples(self, path=None):
        """读取并校验经验快照的棋盘尺寸及坐标、增强版本。恢复历史或 FIFO 缓冲区，拒绝可能包含非法镜像的旧数据。"""
        if path is None:
            model_path = Path(self.args.load_folder_file[0]) / self.args.load_folder_file[1]
            path = Path(str(model_path) + '.examples')
        else:
            path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Replay snapshot not found: {path}. Use --fresh-replay to collect new data.')
        with path.open('rb') as f:
            payload = Unpickler(f).load()
        if not isinstance(payload, dict) or payload.get('augmentation_version') != 2 or payload.get('canonical_version') != 1:
            raise ValueError('Legacy/unversioned replay may contain invalid Hex reflections. Use --fresh-replay to regenerate it.')
        if payload.get('board_size') != self.game.n:
            raise ValueError('Replay board size does not match the game')
        self.trainExamplesHistory = payload['history']
        if self.replayBuffer is not None:
            self.replayBuffer.examples.clear()
            for history in self.trainExamplesHistory:
                self.replayBuffer.extend(history)
            self.trainExamplesHistory = self.replayBuffer.to_history()
        self.skipFirstSelfPlay = False  # 一轮完整结束后再保存快照。

    def _save_snapshot(self, iteration):
        """先保存模型和经验池，再发布最新清单及硬链接别名。只清理清单管理的过期快照，避免误删其他实验数据。"""
        folder = Path(self.args.checkpoint)
        folder.mkdir(parents=True, exist_ok=True)
        model = self.getCheckpointFile(iteration)
        examples = model + '.examples'
        self.nnet.save_checkpoint(str(folder), model)
        self.saveTrainExamples(iteration, filename=examples)
        manifest_path = folder / 'latest.json'
        managed = []
        if manifest_path.exists():
            managed = json.loads(manifest_path.read_text()).get('managed_snapshots', [])
        managed = [entry for entry in managed if entry != model] + [model]
        keep = arg_value(self.args, 'keepCheckpoints', 3)
        manifest = {'iteration': iteration, 'model': model, 'examples': examples,
                    'managed_snapshots': managed[-keep:]}
        # 只发布完整快照对；别名不重复占用经验池空间。
        for alias in ('latest.pth.tar', 'best.pth.tar'):
            atomic_alias(folder / model, folder / alias)
            atomic_alias(folder / examples, folder / (alias + '.examples'))
        fd, temporary = tempfile.mkstemp(dir=folder, prefix='.manifest-')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(manifest, f, indent=2)
            os.replace(temporary, manifest_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        # 仅清理清单中登记的快照，保留其他实验文件。
        for old in managed[:-keep]:
            if Path(old).name != old or not old.startswith('checkpoint_') or not old.endswith('.pth.tar'):
                continue
            for name in (old, old + '.examples'):
                (folder / name).unlink(missing_ok=True)
        log.info('Saved iteration %s; retaining %s managed snapshot pairs', iteration, keep)
