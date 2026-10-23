"""Offline checks and benchmarks for tetris_ai (no browser needed).

1. Key plans: replaying each placement's keys in a tiny SRS simulator lands the piece where planned.
2. T-spin detection on a hand-built T-spin double.
3. Piece identification.
4. Versus self-play with random incoming garbage: attack per piece, survival, decision time.
"""
import random
import time

import tetris_ai as ai
from tetris_ai import COLS, ROWS, SHAPES, SPAWN_X, SPAWN_Y

INV = {v: k for k, v in ai.KEY_NAMES.items()}


def replay(board, piece, keys, x=SPAWN_X, y=SPAWN_Y):
    r = 0
    for k in keys:
        if k == ai.HARD_DROP:
            break
        mv = INV[k]
        if mv in ("L", "R"):
            nx = x + (-1 if mv == "L" else 1)
            if not ai.collides(board, ai._rows(piece, r, nx), y):
                x = nx
        elif mv == "SD":
            while not ai.collides(board, ai._rows(piece, r, x), y + 1):
                y += 1
        elif piece != "O":
            nr = (r + (1 if mv == "CW" else 3)) % 4
            for kx, ky in ai.KICKS[piece][(r, nr)]:
                if not ai.collides(board, ai._rows(piece, nr, x + kx), y + ky):
                    r, x, y = nr, x + kx, y + ky
                    break
    while not ai.collides(board, ai._rows(piece, r, x), y + 1):
        y += 1
    return ai.lock(board, piece, r, x, y)


# ---- 1. key plans -----------------------------------------------------------
rng = random.Random(3)
for _ in range(300):
    board = [0] * ROWS
    for yy in range(ROWS - rng.randint(0, 8), ROWS):
        board[yy] = ai.FULL & ~(1 << rng.randrange(COLS)) & ~(1 << rng.randrange(COLS))
    piece = rng.choice(list(SHAPES))
    for pl in ai.placements(board, piece):
        want = ai.lock(board, piece, pl.rot, pl.x, pl.y)
        assert replay(board, piece, pl.keys()) == want, (piece, pl)
print("key plans OK")

# ---- 2. T-spin double -------------------------------------------------------
g = [[0] * COLS for _ in range(ROWS)]
for c in range(COLS):
    if c not in (3, 4, 5):
        g[18][c] = 1
    if c != 4:
        g[19][c] = 1
g[17][3] = 1  # overhang on the left of the slot
for c in range(3):
    g[17][c] = 1
board = ai.to_bits(g)
tsd = [p for p in ai.placements(board, "T") if p.spin == 2]
assert tsd, "no T-spin found"
new, lines = ai.lock(board, "T", tsd[0].rot, tsd[0].x, tsd[0].y)
assert lines == 2, lines
assert ai._tslots(board) == 1.0
print("T-spin double detected, keys:", tsd[0].keys())

# ---- 3. identification ------------------------------------------------------
for name, rots in SHAPES.items():
    for shift in range(4):
        cells = [(r + 1, c + SPAWN_X + shift) for r, c in rots[0]]
        assert ai.identify_piece(cells) == (name, SPAWN_X + shift)
print("piece identification OK")


# ---- 4. versus self-play ----------------------------------------------------
def old_bot(board, piece, queue, hold, b2b, combo, incoming):
    """First version: El-Tetris, hard drops only, no lookahead (for comparison)."""
    best = None
    for pl in ai.placements(board, piece):
        if any(k in pl.keys() for k in ("ArrowDown",)):
            continue
        new, lines = ai.lock(board, piece, pl.rot, pl.x, pl.y)
        if new is None:
            continue
        hs, holes, _, _, _, _ = ai.features(new)
        bump = sum(abs(a - b) for a, b in zip(hs, hs[1:]))
        s = -0.51 * sum(hs) + 0.76 * lines - 0.36 * holes - 0.18 * bump
        if best is None or s > best[0]:
            best = (s, pl)
    return None if best is None else ai.Decision(False, best[1], best[0])


def new_bot(board, piece, queue, hold, b2b, combo, incoming):
    return ai.search(board, piece, queue, hold, b2b, combo, incoming, depth=3, beam=8)


def play(bot, seed, n=500, garbage_rate=0.35):
    """Opponent sends ~garbage_rate lines per piece in bursts; our attack cancels it first."""
    rng = random.Random(seed)
    board, queue, hold = [0] * ROWS, [], None
    b2b, combo, pending, sent_total, t_think = False, -1, 0, 0, 0.0

    def refill():
        while len(queue) < 7:
            bag = list(SHAPES)
            rng.shuffle(bag)
            queue.extend(bag)

    refill()
    for i in range(n):
        piece = queue.pop(0)
        refill()
        if rng.random() < garbage_rate / 2.5:
            pending += rng.choice((1, 2, 2, 3, 4))
        t0 = time.perf_counter()
        dec = bot(board, piece, queue[:5], hold, b2b, combo, pending)
        t_think += time.perf_counter() - t0
        if dec is None:
            return i, sent_total, t_think / max(i, 1)
        if dec.use_hold:
            if hold is None:
                hold, piece = piece, queue.pop(0)
                refill()
            else:
                hold, piece = piece, hold
        pl = dec.placement
        assert pl.piece == piece
        new, lines = ai.lock(board, piece, pl.rot, pl.x, pl.y)
        if new is None:
            return i, sent_total, t_think / (i + 1)
        pc = lines > 0 and not any(new)
        sent, b2b, combo = ai.attack(lines, pl.spin, b2b, combo, pc)
        cancel = min(sent, pending)
        pending -= cancel
        sent_total += sent - cancel
        board = new
        if lines == 0 and pending:  # garbage enters after a non-clearing placement
            hole = rng.randrange(COLS)
            k = min(pending, 8)
            pending -= k
            if any(board[:k]):
                return i, sent_total, t_think / (i + 1)
            board = board[k:] + [ai.FULL & ~(1 << hole)] * k
    return n, sent_total, t_think / n


for name, bot in (("old (El-Tetris, greedy)", old_bot), ("new (versus search)", new_bot)):
    res = [play(bot, s, n=300) for s in range(2)]
    pieces = sum(r[0] for r in res)
    sent = sum(r[1] for r in res)
    print(f"{name:26s} survived {pieces / 2:5.0f}/300 pieces, "
          f"net attack/piece {sent / pieces:.2f}, think {1000 * sum(r[2] for r in res) / 2:.0f} ms/piece")
