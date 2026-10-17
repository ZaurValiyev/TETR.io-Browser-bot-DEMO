"""Heuristic Tetris placement AI (no browser, no vision: pure logic).

Board: list of ROWS lists of COLS ints (0 empty, 1 filled). Row 0 is the TOP.

Pieces use the SRS bounding-box matrices. In TETR.IO every piece spawns with
its box's left edge at column 3, so the number of left/right taps needed to
reach a target box column is simply (target_x - SPAWN_X).
"""
from __future__ import annotations

ROWS, COLS = 20, 10
SPAWN_X = 3

# SHAPES[piece][rotation] = list of (row, col) cells inside the piece's box.
# Rotation order: 0 = spawn, 1 = clockwise (R), 2 = 180, 3 = counter-clockwise (L).
SHAPES = {
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 1), (1, 2), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
    "I": [
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
    ],
    "O": [[(0, 1), (0, 2), (1, 1), (1, 2)]] * 4,
}

# El-Tetris weights (Yiyuan Lee): a well-known, simple, effective heuristic.
W_HEIGHT, W_LINES, W_HOLES, W_BUMP = -0.510066, 0.760666, -0.35663, -0.184483

# Try cheap rotations first so ties resolve to fewer key presses.
ROTATION_ORDER = (0, 1, 3, 2)

# Key plan for each rotation state (TETR.IO defaults: Up = CW, Z = CCW).
ROTATE_KEYS = {0: [], 1: ["ArrowUp"], 2: ["ArrowUp", "ArrowUp"], 3: ["z"]}


def _collides(board, cells, bx, by):
    for r, c in cells:
        x, y = bx + c, by + r
        if x < 0 or x >= COLS or y >= ROWS:
            return True
        if y >= 0 and board[y][x]:
            return True
    return False


def _hard_drop_y(board, cells, bx):
    """Lowest box row the piece can reach, or None if it can't be placed."""
    by = -3
    if _collides(board, cells, bx, by):
        return None
    while not _collides(board, cells, bx, by + 1):
        by += 1
    return by


def _place(board, cells, bx, by):
    new = [row[:] for row in board]
    for r, c in cells:
        y = by + r
        if y < 0:  # locked above the visible field = top out
            return None, 0
        new[y][bx + c] = 1
    full = [i for i, row in enumerate(new) if all(row)]
    for i in full:
        del new[i]
        new.insert(0, [0] * COLS)
    return new, len(full)


def _features(board):
    heights = [0] * COLS
    holes = 0
    for x in range(COLS):
        seen = False
        for y in range(ROWS):
            if board[y][x]:
                if not seen:
                    heights[x] = ROWS - y
                    seen = True
            elif seen:
                holes += 1
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(COLS - 1))
    return sum(heights), holes, bump


def evaluate(board, lines):
    height, holes, bump = _features(board)
    return W_HEIGHT * height + W_LINES * lines + W_HOLES * holes + W_BUMP * bump


def best_placement(board, piece):
    """Return (rotation, box_x, score) for the best landing spot, or None."""
    best = None
    for rot in ROTATION_ORDER:
        cells = SHAPES[piece][rot]
        for bx in range(-3, COLS):
            by = _hard_drop_y(board, cells, bx)
            if by is None:
                continue
            new, lines = _place(board, cells, bx, by)
            if new is None:
                continue
            score = evaluate(new, lines)
            if best is None or score > best[2] + 1e-9:
                best = (rot, bx, score)
    return best


def _build_spawn_lookup():
    lookup = {}
    for piece, rots in SHAPES.items():
        cells = rots[0]
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        norm = frozenset((r - mr, c - mc) for r, c in cells)
        lookup[norm] = (piece, mc)
    return lookup


_SPAWN_LOOKUP = _build_spawn_lookup()


def identify_piece(cells):
    """Identify a freshly spawned piece from its 4 filled (row, col) cells.

    Works from the cell pattern alone (no colours), so it is skin-independent.
    Returns (piece, current_box_x) or None if the pattern is not a spawn shape.
    """
    if len(cells) != 4:
        return None
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    hit = _SPAWN_LOOKUP.get(frozenset((r - mr, c - mc) for r, c in cells))
    if hit is None:
        return None
    piece, offset = hit
    return piece, mc - offset


def key_plan(rotation, box_x, start_x=SPAWN_X):
    """Keys to press, in order, to execute a placement from the current state."""
    keys = list(ROTATE_KEYS[rotation])
    dx = box_x - start_x
    keys += (["ArrowRight"] if dx > 0 else ["ArrowLeft"]) * abs(dx)
    keys.append("Space")  # hard drop
    return keys


def apply_placement(board, piece, rotation, box_x):
    """Simulate a placement (used by the offline test). Returns (board, lines)."""
    cells = SHAPES[piece][rotation]
    by = _hard_drop_y(board, cells, box_x)
    return _place(board, cells, box_x, by)
