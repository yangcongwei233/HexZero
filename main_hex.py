import argparse
import logging
import os
import json
import random
import sys

import numpy as np
import torch

try:
    import coloredlogs
except ImportError:
    coloredlogs = None

from Coach import Coach
from hex.HexGame import HexGame as Game
from hex.pytorch.NNet import args as nnet_args
from hex.pytorch.NNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)

if coloredlogs:
    coloredlogs.install(level='INFO')
else:
    logging.basicConfig(level=logging.INFO)


PRESETS = {
    'smoke': {
        'numIters': 2,
        'numEps': 4,
        'tempThreshold': 8,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 20000,
        'numMCTSSims': 10,
        'arenaCompare': 4,
        'numItersForTrainExamplesHistory': 5,
    },
    'longer': {
        'numIters': 30,
        'numEps': 50,
        'tempThreshold': 12,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 50,
        'arenaCompare': 20,
        'numItersForTrainExamplesHistory': 20,
    },
}


def build_args(cli_args):
    preset = PRESETS[cli_args.preset].copy()
    checkpoint = cli_args.checkpoint_dir or './temp/hex_{}x{}/'.format(cli_args.board_size, cli_args.board_size)

    overrides = {
        'numIters': cli_args.num_iters,
        'numEps': cli_args.num_eps,
        'tempThreshold': cli_args.temp_threshold,
        'updateThreshold': cli_args.update_threshold,
        'maxlenOfQueue': cli_args.maxlen_of_queue,
        'numMCTSSims': cli_args.num_mcts_sims,
        'arenaCompare': cli_args.arena_compare,
        'numItersForTrainExamplesHistory': cli_args.history_iters,
    }
    for key, value in overrides.items():
        if value is not None:
            preset[key] = value

    preset.update({
        'cpuct': cli_args.cpuct,
        'checkpoint': checkpoint,
        'load_model': cli_args.load_model,
        'load_folder_file': (checkpoint, cli_args.checkpoint_file),
        'startIter': cli_args.start_iter or 1,
        'useOpeningBook': cli_args.use_opening_book,
        'useTacticalCheck': cli_args.use_tactical_check,
        'heuristicPriorWeight': cli_args.heuristic_prior_weight,
        'parallelSelfPlay': cli_args.parallel_self_play,
        'numWorkers': cli_args.num_workers,
        'gamesPerWorker': cli_args.games_per_worker,
        'arenaWorkers': cli_args.arena_workers,
        'arenaDevice': cli_args.arena_device,
        'arenaThreads': cli_args.arena_threads,
        'mctsBackend': cli_args.mcts_backend,
        'mctsBatchSize': cli_args.mcts_batch_size,
        'reuseSearchTree': cli_args.reuse_search_tree,
        'winCheckMode': cli_args.win_check_mode,
        'selfplayDevice': cli_args.selfplay_device,
        'replayBufferSize': cli_args.replay_buffer_size,
        'teacherData': cli_args.teacher_data,
        'teacherFraction': cli_args.teacher_fraction,
        'rootDirichletNoise': cli_args.root_dirichlet_noise,
        'dirichletAlpha': cli_args.dirichlet_alpha,
        'dirichletFraction': cli_args.dirichlet_fraction,
        'saveSelfplayJsonl': cli_args.save_selfplay_jsonl,
        'selfplayOutputDir': cli_args.selfplay_output_dir,
        'savePolicyInJsonl': cli_args.save_policy_in_jsonl,
        'debugValueDiagnostics': cli_args.debug_value_diagnostics,
        'seed': cli_args.seed if cli_args.seed is not None else 0,
        'selfplayThreads': cli_args.selfplay_threads,
        'checkpointInterval': cli_args.checkpoint_interval,
        'keepCheckpoints': cli_args.keep_checkpoints,
        'tensorboard': cli_args.tensorboard or cli_args.tensorboard_dir is not None,
        'tensorboardDir': cli_args.tensorboard_dir,
    })
    return dotdict(preset)


def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero-style Hex.')
    parser.add_argument('--board-size', type=int, default=5, help='Use 5 for quick runs, later 7 or 11.')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='smoke')
    parser.add_argument('--load-model', action='store_true', help='Continue from checkpoint_file in the board checkpoint folder.')
    parser.add_argument('--checkpoint-dir', default=None,
                        help='Directory for checkpoints/examples. Use a new directory when retraining with a changed architecture.')
    parser.add_argument('--checkpoint-file', default='latest.pth.tar')
    parser.add_argument('--examples-file', default=None, help='Load train examples from this file in the checkpoint folder.')
    parser.add_argument('--skip-first-self-play', action='store_true',
                        help='Use when examples_file was collected for an interrupted iteration that has not trained yet.')
    parser.add_argument('--start-iter', type=int, default=None, help='Defaults to the iteration after the loaded checkpoint.')
    parser.add_argument('--num-iters', type=int, default=None)
    parser.add_argument('--num-eps', type=int, default=None)
    parser.add_argument('--num-mcts-sims', type=int, default=None)
    parser.add_argument('--arena-compare', type=int, default=None)
    parser.add_argument('--temp-threshold', type=int, default=None)
    parser.add_argument('--update-threshold', type=float, default=None)
    parser.add_argument('--maxlen-of-queue', type=int, default=None)
    parser.add_argument('--history-iters', type=int, default=None)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--use-opening-book', action=argparse.BooleanOptionalAction, default=False,
                        help='Use Hex opening book before MCTS. Default is off for training.')
    parser.add_argument('--use-tactical-check', action=argparse.BooleanOptionalAction, default=False,
                        help='Use Hex tactical checks before MCTS. Default is off for training.')
    parser.add_argument('--heuristic-prior-weight', type=float, default=0.0,
                        help='Blend heuristic Hex prior into MCTS leaf policy. 0.0 preserves original behavior.')
    parser.add_argument('--parallel-self-play', action=argparse.BooleanOptionalAction, default=False,
                        help='Generate self-play games in worker processes.')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel self-play worker processes.')
    parser.add_argument('--games-per-worker', type=int, default=None,
                        help='Self-play games per worker task. Defaults to an even split of num_eps.')
    parser.add_argument('--arena-workers', type=int, default=1,
                        help='Parallel evaluation processes; 1 uses the serial Arena.')
    parser.add_argument('--arena-device', choices=('cpu', 'cuda'), default=None,
                        help='Parallel evaluation device. Defaults to the training network device.')
    parser.add_argument('--arena-threads', type=int, default=1,
                        help='Torch CPU threads per evaluation worker.')
    parser.add_argument('--mcts-backend', choices=('classic', 'azalea'), default='classic',
                        help='MCTS implementation. classic preserves the original alpha-zero-general backend.')
    parser.add_argument('--mcts-batch-size', type=int, default=16,
                        help='Leaf evaluation batch size for --mcts-backend azalea.')
    parser.add_argument('--reuse-search-tree', action=argparse.BooleanOptionalAction, default=False,
                        help='Reuse azalea search tree after executed moves when safe.')
    parser.add_argument('--root-dirichlet-noise', action=argparse.BooleanOptionalAction, default=False,
                        help='Add Dirichlet noise to the azalea root prior.')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet-fraction', type=float, default=0.25)
    parser.add_argument('--win-check-mode', choices=('bfs', 'fast', 'numba'), default='numba',
                        help='Hex win check implementation. numba falls back to BFS when unavailable.')
    parser.add_argument('--selfplay-device', choices=('cpu', 'cuda'), default='cpu',
                        help='Device used inside parallel self-play workers.')
    parser.add_argument('--replay-buffer-size', type=int, default=0,
                        help='If >0, keep a FIFO replay buffer with this many examples.')
    parser.add_argument('--teacher-data', default=None, help='Versioned Azalea teacher .npz; kept separate from FIFO replay.')
    parser.add_argument('--teacher-fraction', type=float, default=0.,
                        help='Fraction of each training batch sampled from teacher data; e.g. 0.1. Default is off.')
    parser.add_argument('--num-iters-for-train-examples-history', dest='history_iters', type=int, default=None,
                        help='Alias for --history-iters.')
    parser.add_argument('--save-selfplay-jsonl', action='store_true')
    parser.add_argument('--selfplay-output-dir', default=None)
    parser.add_argument('--save-policy-in-jsonl', action='store_true')
    parser.add_argument('--disable-value-bn', action='store_true',
                        help='Replace the single-channel value BatchNorm with Identity for ablation.')
    parser.add_argument('--value-channels', type=int, default=None,
                        help='Value head channels. Default is 8; use 2 for lower compute.')
    parser.add_argument('--num-res-blocks', type=int, default=None,
                        help='Residual trunk depth. Default is 3; azalea Hex uses 6.')
    parser.add_argument('--allow-partial-checkpoint', action='store_true',
                        help='Load only shape-compatible checkpoint tensors. Use for architecture migration, not normal resume.')
    parser.add_argument('--debug-value-diagnostics', action='store_true',
                        help='Enable slower value target/head diagnostics for debugging.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed; resume defaults to the saved seed.')
    parser.add_argument('--selfplay-threads', type=int, default=1, help='Torch CPU threads per worker.')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save a model/replay pair every N iterations and on normal completion.')
    parser.add_argument('--keep-checkpoints', type=int, default=3, help='Retain this many newly managed checkpoint/replay pairs.')
    parser.add_argument('--fresh-replay', action='store_true', help='Load weights/training state but regenerate all self-play data.')
    parser.add_argument('--tensorboard', action='store_true', help='Write TensorBoard training, self-play and Arena metrics.')
    parser.add_argument('--tensorboard-dir', default=None, help='Enables TensorBoard; defaults to CHECKPOINT_DIR/tensorboard.')
    parser.add_argument('--tensorboard-interval', type=int, default=10, help='Log minibatch losses every N optimizer steps.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    cli = parser.parse_args()
    positive = ('board_size', 'num_iters', 'num_eps', 'num_mcts_sims', 'num_workers',
                'mcts_batch_size', 'maxlen_of_queue', 'history_iters', 'value_channels',
                'checkpoint_interval', 'keep_checkpoints', 'selfplay_threads',
                'tensorboard_interval', 'epochs', 'batch_size', 'start_iter', 'games_per_worker', 'num_res_blocks',
                'arena_workers', 'arena_threads')
    for name in positive:
        value = getattr(cli, name)
        if value is not None and value < 1:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    if cli.arena_compare is not None and (cli.arena_compare < 2 or cli.arena_compare % 2):
        parser.error('--arena-compare must be a positive even number')
    if cli.seed is not None and cli.seed < 0:
        parser.error('--seed must be nonnegative')
    if cli.replay_buffer_size < 0:
        parser.error('--replay-buffer-size must be nonnegative')
    if not 0 <= cli.teacher_fraction < 1:
        parser.error('--teacher-fraction must be in [0, 1)')
    if cli.learning_rate is not None and cli.learning_rate <= 0:
        parser.error('--learning-rate must be positive')
    if cli.cpuct < 0 or cli.dirichlet_alpha <= 0 or not 0 <= cli.dirichlet_fraction <= 1:
        parser.error('Invalid search exploration parameters')
    if not 0 <= cli.heuristic_prior_weight <= 1 or (cli.update_threshold is not None and not 0 <= cli.update_threshold <= 1):
        parser.error('Prior weight and update threshold must be in [0, 1]')
    if cli.fresh_replay and (cli.examples_file or cli.skip_first_self_play):
        parser.error('--fresh-replay cannot be combined with loaded replay or skipping self-play')
    if (cli.examples_file or cli.skip_first_self_play or cli.fresh_replay) and not cli.load_model:
        parser.error('Replay resume options require --load-model')
    cli.explicit_options = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    return cli


RESUME_OPTIONS = {
    'numEps': ('--num-eps', '--preset'),
    'tempThreshold': ('--temp-threshold', '--preset'),
    'updateThreshold': ('--update-threshold', '--preset'),
    'maxlenOfQueue': ('--maxlen-of-queue', '--preset'),
    'numMCTSSims': ('--num-mcts-sims', '--preset'),
    'arenaCompare': ('--arena-compare', '--preset'),
    'numItersForTrainExamplesHistory': ('--history-iters', '--num-iters-for-train-examples-history', '--preset'),
    'cpuct': ('--cpuct',), 'useOpeningBook': ('--use-opening-book',),
    'useTacticalCheck': ('--use-tactical-check',), 'heuristicPriorWeight': ('--heuristic-prior-weight',),
    'parallelSelfPlay': ('--parallel-self-play',), 'numWorkers': ('--num-workers',),
    'gamesPerWorker': ('--games-per-worker',), 'selfplayThreads': ('--selfplay-threads',),
    'arenaWorkers': ('--arena-workers',), 'arenaDevice': ('--arena-device',),
    'arenaThreads': ('--arena-threads',),
    'mctsBackend': ('--mcts-backend',), 'mctsBatchSize': ('--mcts-batch-size',),
    'reuseSearchTree': ('--reuse-search-tree',), 'rootDirichletNoise': ('--root-dirichlet-noise',),
    'dirichletAlpha': ('--dirichlet-alpha',), 'dirichletFraction': ('--dirichlet-fraction',),
    'selfplayDevice': ('--selfplay-device',), 'replayBufferSize': ('--replay-buffer-size',),
    'teacherData': ('--teacher-data',), 'teacherDataSha256': ('--teacher-data',),
    'teacherFraction': ('--teacher-fraction',),
    'checkpointInterval': ('--checkpoint-interval',), 'keepCheckpoints': ('--keep-checkpoints',),
}


def main():
    cli_args = parse_args()
    args = build_args(cli_args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log.info('Loading %s...', Game.__name__)
    g = Game(cli_args.board_size, win_check_mode=cli_args.win_check_mode)

    log.info('Loading %s...', nn.__name__)
    if cli_args.value_channels is not None:
        nnet_args.value_channels = cli_args.value_channels
    if cli_args.num_res_blocks is not None:
        nnet_args.num_res_blocks = cli_args.num_res_blocks
    nnet_args.disable_value_bn = cli_args.disable_value_bn
    nnet_args.allow_partial_checkpoint = cli_args.allow_partial_checkpoint
    nnet_args.debug_value_diagnostics = cli_args.debug_value_diagnostics
    nnet_args.tensorboard_interval = cli_args.tensorboard_interval
    nnet = nn(g)
    log.info('Hex training device: %s', 'CUDA' if nnet_args.cuda else 'CPU')
    log.info('Training config: board=%sx%s preset=%s iters=%s eps=%s sims=%s arena=%s parallel=%s workers=%s backend=%s batch=%s policy_channels=%s value_channels=%s disable_value_bn=%s',
             cli_args.board_size, cli_args.board_size, cli_args.preset, args.numIters,
             args.numEps, args.numMCTSSims, args.arenaCompare, args.parallelSelfPlay, args.numWorkers,
             args.mctsBackend, args.mctsBatchSize, nnet_args.policy_channels, nnet_args.value_channels,
             nnet_args.disable_value_bn)

    if args.load_model:
        manifest_path = os.path.join(args.checkpoint, 'latest.json')
        if cli_args.checkpoint_file in ('latest.pth.tar', 'best.pth.tar') and os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            args.load_folder_file = (args.checkpoint, manifest['model'])
        log.info('Loading checkpoint "%s/%s"...', *args.load_folder_file)
        nnet.load_checkpoint(*args.load_folder_file, restore_rng=True)
        metadata = nnet.checkpoint_metadata
        saved_config = metadata.get('run_config', {})
        for name, options in RESUME_OPTIONS.items():
            if name in saved_config and not cli_args.explicit_options.intersection(options + tuple('--no-' + option[2:] for option in options)):
                args[name] = saved_config[name]
        if cli_args.value_channels is not None and nnet.args.value_channels != cli_args.value_channels:
            raise ValueError('Requested value channels differ from checkpoint; use --allow-partial-checkpoint for migration')
        if cli_args.num_res_blocks is not None and nnet.args.num_res_blocks != cli_args.num_res_blocks:
            raise ValueError('Requested residual depth differs from checkpoint; use --allow-partial-checkpoint for migration')
        if cli_args.disable_value_bn and not nnet.args.disable_value_bn:
            raise ValueError('Requested value BN removal differs from checkpoint; use --allow-partial-checkpoint for migration')
        if cli_args.start_iter is None:
            if 'iteration' not in metadata:
                raise ValueError('Legacy checkpoint has no iteration metadata; specify --start-iter explicitly')
            args.startIter = int(metadata['iteration']) + 1
        if cli_args.seed is None:
            args.seed = int(metadata.get('seed', args.seed))
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
    else:
        log.warning('Starting from a new network')
    # 迁移只在加载时执行一次，不作为评估失败后的回滚策略。
    nnet.args.allow_partial_checkpoint = False
    nnet_args.allow_partial_checkpoint = False
    for cli_name, net_name in (('epochs', 'epochs'), ('batch_size', 'batch_size'), ('learning_rate', 'lr')):
        value = getattr(cli_args, cli_name)
        if value is not None:
            nnet.args[net_name] = value
    if cli_args.learning_rate is not None:
        for group in nnet.optimizer.param_groups:
            group['lr'] = cli_args.learning_rate
    log.info('Loading the Coach...')
    log.info('Evaluation config: games=%s workers=%s device=%s threads=%s',
             args.arenaCompare, args.arenaWorkers,
             args.arenaDevice or ('cuda' if nnet.args.cuda else 'cpu'), args.arenaThreads)
    c = Coach(g, nnet, args)
    if args.load_model and not cli_args.fresh_replay:
        examples_path = os.path.join(args.checkpoint, cli_args.examples_file) if cli_args.examples_file else None
        c.loadTrainExamples(examples_path)
        c.skipFirstSelfPlay = cli_args.skip_first_self_play

    log.info('Starting Hex learning process')
    c.learn()


if __name__ == "__main__":
    main()
