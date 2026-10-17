"""Offline self-play: checks the AI and key-plan logic without any browser."""
import random

from tetris_ai import COLS, ROWS, SHAPES, SPAWN_X, apply_placement, best_placement, key_plan

random.seed(1)
board = [[0] * COLS for _ in range(ROWS)]
total_lines, pieces = 0, 0
bag = []
while pieces < 2000:
    if not bag:
        bag = list(SHAPES)
        random.shuffle(bag)
    piece = bag.pop()
    move = best_placement(board, piece)
    if move is None:
        break
    rot, bx, _ = move
    keys = key_plan(rot, bx)
    assert keys[-1] == "space"
    # key plan must reproduce the target column from the spawn column
    dx = keys.count("ArrowRight") - keys.count("ArrowLeft")
    assert SPAWN_X + dx == bx
    board, lines = apply_placement(board, piece, rot, bx)
    total_lines += lines
    pieces += 1

print(f"survived {pieces} pieces, cleared {total_lines} lines")
