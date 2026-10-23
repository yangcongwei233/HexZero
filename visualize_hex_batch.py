import argparse
import csv
import os

import imageio.v2 as imageio
import numpy as np

from MCTS import MCTS
from hex.HexGame import HexGame
from hex.HexLogic import action_to_coord
from hex.pytorch.NNet import NNetWrapper
from utils import dotdict
from visualize_hex import draw_board


def select_action(pi, temp):
    if temp == 0:
        return int(np.argmax(pi))

    probs = np.asarray(pi, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def play_game(game, nnet, num_mcts_sims, sample_moves):
    args = dotdict({"numMCTSSims": num_mcts_sims, "cpuct": 1})
    board = game.getInitBoard()
    player = 1
    history = [(np.copy(board), None, 0, None, None)]
    moves = []

    for move_number in range(1, game.getActionSize() + 1):
        mcts = MCTS(game, nnet, args)
        canonical = game.getCanonicalForm(board, player)
        temp = int(move_number <= sample_moves)
        pi = mcts.getActionProb(canonical, temp=temp)
        canonical_action = select_action(pi, temp=temp)
        action = game.actionFromCanonical(canonical_action, player)
        row, col = action_to_coord(action, game.n)
        moves.append((move_number, player, row, col, action))

        board, player = game.getNextState(board, player, action)
        result_for_next_player = game.getGameEnded(board, player)
        winner = None
        if result_for_next_player != 0:
            winner = -player if result_for_next_player == -1 else player

        history.append((np.copy(board), (row, col), move_number, winner, action))
        if winner is not None:
            break

    return history, moves


def save_game(out_dir, game_index, history, moves):
    game_dir = os.path.join(out_dir, "game_{:02d}".format(game_index))
    os.makedirs(game_dir, exist_ok=True)

    frame_paths = []
    for board, last_move, move_number, winner, _ in history:
        frame_path = os.path.join(game_dir, "move_{:02d}.png".format(move_number))
        draw_board(board, frame_path, last_move=last_move, move_number=move_number, winner=winner)
        frame_paths.append(frame_path)

    gif_path = os.path.join(game_dir, "game_{:02d}.gif".format(game_index))
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, frames, duration=0.65)

    final_png = frame_paths[-1]
    moves_path = os.path.join(game_dir, "moves.csv")
    with open(moves_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["move", "player", "row", "col", "action"])
        writer.writerows(moves)

    winner = history[-1][3]
    return {
        "game": game_index,
        "winner": "P1" if winner == 1 else "P2",
        "moves": len(moves),
        "gif": gif_path,
        "final_png": final_png,
        "moves_csv": moves_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run and visualize multiple Hex model self-play games.")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-games", type=int, default=6)
    parser.add_argument("--num-mcts-sims", type=int, default=100)
    parser.add_argument("--sample-moves", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out-dir", default="./temp/hex_5x5/visualization_latest")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    game = HexGame(args.board_size, use_fast_win_check=True)
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(os.path.dirname(args.checkpoint), os.path.basename(args.checkpoint))

    summaries = []
    for game_index in range(1, args.num_games + 1):
        print("Running game {}/{}...".format(game_index, args.num_games), flush=True)
        history, moves = play_game(game, nnet, args.num_mcts_sims, args.sample_moves)
        summaries.append(save_game(args.out_dir, game_index, history, moves))

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "winner", "moves", "gif", "final_png", "moves_csv"])
        writer.writeheader()
        writer.writerows(summaries)

    summary_txt = os.path.join(args.out_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("checkpoint: {}\n".format(args.checkpoint))
        f.write("num_games: {}\n".format(args.num_games))
        f.write("num_mcts_sims: {}\n".format(args.num_mcts_sims))
        f.write("sample_moves: {}\n".format(args.sample_moves))
        f.write("seed: {}\n\n".format(args.seed))
        for row in summaries:
            f.write("game {game}: winner={winner}, moves={moves}, gif={gif}\n".format(**row))

    print("Saved summary to {}".format(summary_txt))


if __name__ == "__main__":
    main()
