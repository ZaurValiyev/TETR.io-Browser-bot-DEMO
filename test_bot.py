"""Offline checks for bot.py's sync logic (no browser)."""
import random
import bot
import tetris_ai as ai
from tetris_ai import ROWS, COLS, FULL

rng = random.Random(1)
ok = 0
for _ in range(500):
    sim = [0]*ROWS
    for y in range(ROWS - rng.randint(0, 9), ROWS):
        sim[y] = FULL & ~(1 << rng.randrange(COLS)) & ~(1 << rng.randrange(COLS))
    piece = rng.choice("TSZJLIO")
    k = rng.randint(0, 5)
    screen = sim[k:] + [FULL & ~(1 << rng.randrange(COLS)) for _ in range(k)]
    gh = bot.landing_cells(screen, piece, 3)   # ghost drawn at the REAL (risen) spot
    vis = list(screen)
    for y, m in gh or ():
        vis[y] |= m
    for y in range(bot.TOP_MASK_ROWS):
        vis[y] = 0
    b, rise = bot.match_board(sim, vis, piece, 3)
    assert b is not None and rise == k and b[bot.TOP_MASK_ROWS:] == screen[bot.TOP_MASK_ROWS:], (k, piece)
    ok += 1
# a real mismatch must not be accepted
sim = [0]*ROWS; sim[19] = FULL & ~1
vis = [0]*ROWS; vis[19] = FULL & ~2
assert bot.match_board(sim, vis, "O", 3)[0] is None
print("garbage rise detection OK", ok)

# path poses: lag on the path is recognised
board = [0]*ROWS
pl = [p for p in ai.placements(board, "J") if p.rot == 1 and p.x == 7][0]
path = bot.path_poses(pl)
assert (0, 3) in path and (1, 7) in path and (3, 0) not in path
print("path poses OK", sorted(path))
