import argparse
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

from MCTS import MCTS
from hex.HexGame import HexGame
from hex.HexLogic import action_to_coord
from hex.pytorch.NNet import NNetWrapper
from utils import dotdict


COLORS = {
    0: "#f8f7f2",
    1: "#2f80ed",
    -1: "#eb5757",
}


def hex_center(row, col, y_scale):
    """将棋盘行列转换为倾斜六角网格的绘图中心坐标。"""
    return col + row * 0.5, -row * y_scale


def draw_goal_edges(ax, n, radius, y_scale):
    """在绘图坐标中绘制双方目标边，沿用该旧可视化工具的配色。"""
    top_left = hex_center(0, 0, y_scale)
    top_right = hex_center(0, n - 1, y_scale)
    bottom_left = hex_center(n - 1, 0, y_scale)
    bottom_right = hex_center(n - 1, n - 1, y_scale)

    blue_style = dict(color=COLORS[1], linewidth=7, solid_capstyle="round", alpha=0.9)
    red_style = dict(color=COLORS[-1], linewidth=7, solid_capstyle="round", alpha=0.9)

    ax.plot([top_left[0] - radius * 0.85, top_right[0] + radius * 0.85],
            [top_left[1] + radius * 1.05, top_right[1] + radius * 1.05], **blue_style)
    ax.plot([bottom_left[0] - radius * 0.85, bottom_right[0] + radius * 0.85],
            [bottom_left[1] - radius * 1.05, bottom_right[1] - radius * 1.05], **blue_style)

    ax.plot([top_left[0] - radius * 1.0, bottom_left[0] - radius * 1.0],
            [top_left[1] + radius * 0.55, bottom_left[1] - radius * 0.55], **red_style)
    ax.plot([top_right[0] + radius * 1.0, bottom_right[0] + radius * 1.0],
            [top_right[1] + radius * 0.55, bottom_right[1] - radius * 0.55], **red_style)


def draw_board(board, path, last_move=None, move_number=0, winner=None):
    """绘制指定局面、上一手和结果标记，保存静态图片后释放绘图资源。"""
    n = board.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))
    ax.set_aspect("equal")
    ax.axis("off")

    radius = 0.52
    y_scale = np.sqrt(3) / 2
    draw_goal_edges(ax, n, radius, y_scale)

    for row in range(n):
        for col in range(n):
            x, y = hex_center(row, col, y_scale)
            stone = int(board[row, col])
            patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=radius,
                orientation=np.radians(30),
                facecolor=COLORS[stone],
                edgecolor="#333333",
                linewidth=1.2,
            )
            ax.add_patch(patch)

            if stone != 0:
                text_color = "white"
                ax.text(x, y, "X" if stone == 1 else "O", ha="center", va="center",
                        color=text_color, fontsize=13, fontweight="bold")

            if last_move == (row, col):
                marker = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=radius * 0.78,
                    orientation=np.radians(30),
                    facecolor="none",
                    edgecolor="#111111",
                    linewidth=2.6,
                )
                ax.add_patch(marker)

    ax.text(-0.8, 0.6, "P1 top-bottom", color=COLORS[1], fontsize=11, fontweight="bold")
    ax.text(n - 0.4 + n * 0.5, -n * y_scale + 0.05, "P2 left-right",
            color=COLORS[-1], fontsize=11, fontweight="bold", ha="right")

    title = "Hex self-play, move {}".format(move_number)
    if winner is not None:
        title += " | winner: {}".format("P1" if winner == 1 else "P2")
    ax.set_title(title, fontsize=14)

    ax.set_xlim(-1.2, n + n * 0.5 + 0.2)
    ax.set_ylim(-n * y_scale - 0.7, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def play_game(game, nnet, num_mcts_sims):
    """使用经典 MCTS 完成一局模型自博弈，保存每步棋盘历史供逐帧绘图。"""
    args = dotdict({"numMCTSSims": num_mcts_sims, "cpuct": 1})
    board = game.getInitBoard()
    player = 1
    history = [(np.copy(board), None, 0, None)]

    for move_number in range(1, game.getActionSize() + 1):
        mcts = MCTS(game, nnet, args)
        canonical = game.getCanonicalForm(board, player)
        pi = mcts.getActionProb(canonical, temp=0)
        canonical_action = int(np.argmax(pi))
        action = game.actionFromCanonical(canonical_action, player)
        row, col = action_to_coord(action, game.n)
        board, player = game.getNextState(board, player, action)

        result_for_next_player = game.getGameEnded(board, player)
        winner = None
        if result_for_next_player != 0:
            winner = -player if result_for_next_player == -1 else player

        history.append((np.copy(board), (row, col), move_number, winner))
        if winner is not None:
            break

    return history


def main():
    """加载指定模型或初始化网络，执行一局并导出棋盘帧及动画。"""
    parser = argparse.ArgumentParser(description="Visualize a Hex model self-play game.")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-mcts-sims", type=int, default=25)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gif-name", default="hex_selfplay.gif")
    args = parser.parse_args()

    checkpoint = args.checkpoint or "./temp/hex_{0}x{0}/best.pth.tar".format(args.board_size)
    out_dir = args.out_dir or "./temp/hex_{0}x{0}/visualization".format(args.board_size)
    os.makedirs(out_dir, exist_ok=True)

    game = HexGame(args.board_size, use_fast_win_check=True)
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(os.path.dirname(checkpoint), os.path.basename(checkpoint))

    history = play_game(game, nnet, args.num_mcts_sims)
    frame_paths = []
    for board, last_move, move_number, winner in history:
        frame_path = os.path.join(out_dir, "move_{:02d}.png".format(move_number))
        draw_board(board, frame_path, last_move=last_move, move_number=move_number, winner=winner)
        frame_paths.append(frame_path)

    gif_path = os.path.join(out_dir, args.gif_name)
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, frames, duration=0.75)

    print("Saved {} frames to {}".format(len(frame_paths), out_dir))
    print("Saved GIF to {}".format(gif_path))


if __name__ == "__main__":
    main()
