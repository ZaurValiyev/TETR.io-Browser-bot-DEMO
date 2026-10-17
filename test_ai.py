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
    assert keys[-1] == "Space"
    # key plan must reproduce the target column from the spawn column
    dx = keys.count("ArrowRight") - keys.count("ArrowLeft")
    assert SPAWN_X + dx == bx
    board, lines = apply_placement(board, piece, rot, bx)
    total_lines += lines
    pieces += 1

print(f"survived {pieces} pieces, cleared {total_lines} lines")

# Shape-based piece detection: every piece at spawn (and shifted) must be identified.
from tetris_ai import identify_piece

for name, rots in SHAPES.items():
    for shift in range(0, 4):
        cells = [(r + 1, c + SPAWN_X + shift) for r, c in rots[0]]  # drawn 1 row down
        got = identify_piece(cells)
        assert got == (name, SPAWN_X + shift), (name, shift, got)
print("piece identification OK for all 7 pieces")
