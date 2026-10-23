#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hex.HexGame import HexGame
from hex.pytorch.HexNNet import HexNNet
from hex.pytorch.NNet import args as nnet_args
from hex.value_diagnostics import format_numeric_stats, value_stats


log = logging.getLogger(__name__)


def make_board(n, positive, variant):
    board = np.zeros((n, n), dtype=np.float32)
    offset = variant % n

    if positive:
        board[:, offset] = 1
        board[variant % n, (offset + 2) % n] = -1
        board[(variant + 2) % n, (offset + 3) % n] = -1
    else:
        row = offset
        board[row, :] = -1
        board[(row + 2) % n, variant % n] = 1
        board[(row + 3) % n, (variant + 1) % n] = 1

    if variant % 2 == 1:
        board = np.flipud(board)
    if variant % 3 == 1:
        board = np.fliplr(board)
    return np.ascontiguousarray(board)


def make_dataset(n, repeats):
    boards = []
    values = []
    for variant in range(repeats):
        boards.append(make_board(n, positive=True, variant=variant))
        values.append(1.0)
        boards.append(make_board(n, positive=False, variant=variant))
        values.append(-1.0)
    return np.asarray(boards, dtype=np.float32), np.asarray(values, dtype=np.float32)


def summarize_predictions(label, preds, targets):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    pos = preds[targets > 0]
    neg = preds[targets < 0]
    log.info("%s all: %s", label, format_numeric_stats(value_stats(preds)))
    log.info("%s target=+1: %s", label, format_numeric_stats(value_stats(pos)))
    log.info("%s target=-1: %s", label, format_numeric_stats(value_stats(neg)))


def log_feature_debug(model, boards):
    model.train()
    with torch.no_grad():
        _, _, debug = model.forward_debug(boards)
    for label in ('trunk', 'value_conv', 'value_bn', 'value_fc1', 'value_fc2_tanh'):
        values = debug[label].detach().cpu().numpy()
        log.info("feature %s: %s", label, format_numeric_stats(value_stats(values)))
    policy_logits = debug['policy_logits'].detach().cpu().numpy()
    log.info("feature policy_logits std=%.6f", float(policy_logits.std()))


def evaluate(model, boards, targets, label):
    model.eval()
    with torch.no_grad():
        _, pred_v = model(boards)
    summarize_predictions(label, pred_v.detach().cpu().numpy(), targets.detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description='Debug whether the Hex value head can learn a tiny dataset.')
    parser.add_argument('--board-size', type=int, default=5)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dataset-repeats', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable-value-bn', action='store_true')
    parser.add_argument('--checkpoint', default=None,
                        help='Optional checkpoint path to inspect before the tiny learning run.')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    nnet_args.disable_value_bn = args.disable_value_bn
    nnet_args.cuda = torch.cuda.is_available() and not args.cpu and os.environ.get('HEX_FORCE_CPU') != '1'

    game = HexGame(args.board_size)
    model = HexNNet(game, nnet_args)
    device = torch.device('cuda' if nnet_args.cuda else 'cpu')
    model.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        skipped = sorted(set(state_dict) - set(compatible_state))
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        log.info(
            "loaded checkpoint=%s missing_keys=%s unexpected_keys=%s skipped_incompatible_keys=%s",
            args.checkpoint,
            missing,
            unexpected,
            skipped,
        )

    boards_np, targets_np = make_dataset(args.board_size, args.dataset_repeats)
    boards = torch.tensor(boards_np, dtype=torch.float32, device=device)
    targets = torch.tensor(targets_np, dtype=torch.float32, device=device)

    log.info(
        "debug value learning: board=%s examples=%s disable_value_bn=%s device=%s",
        args.board_size,
        len(boards_np),
        args.disable_value_bn,
        device,
    )
    log_feature_debug(model, boards[:args.batch_size])
    evaluate(model, boards, targets, 'before')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        sample_ids = np.random.randint(len(boards_np), size=min(args.batch_size, len(boards_np)))
        batch_boards = boards[sample_ids]
        batch_targets = targets[sample_ids]

        model.train()
        _, pred_v = model(batch_boards)
        loss = F.mse_loss(pred_v.float().view(-1), batch_targets.float().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 4) == 0:
            log.info("step=%s loss_v=%.6f pred_v %s",
                     step,
                     float(loss.item()),
                     format_numeric_stats(value_stats(pred_v.detach().cpu().numpy())))

    evaluate(model, boards, targets, 'after')


if __name__ == '__main__':
    main()
