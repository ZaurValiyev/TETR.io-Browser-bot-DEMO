"""Versus-oriented Tetris engine (pure logic: no browser, no vision).

What changed compared with the first version:
  * Board is a list of ROWS bitmask ints (row 0 = TOP, bit c = column c) -> much faster.
  * Real movement search: BFS over left / right / rotate (SRS kicks) / soft drop,
    so the bot finds tucks, slides and T-spins, not only "rotate, shift, hard drop".
  * T-spin detection (3-corner rule, mini vs full), back-to-back, combo, perfect clear.
  * Beam search over the NEXT queue with hold.
  * Versus evaluation (Cold Clear-style feature set) instead of El-Tetris survival weights.

Keys assume TETR.IO handling: SDF = infinite (one Down press = instant soft drop),
Up = rotate CW, Z = rotate CCW, C = hold, Space = hard drop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

ROWS, COLS = 20, 10
FULL = (1 << COLS) - 1
SPAWN_X, SPAWN_Y = 3, -2  # piece box top-left at spawn; y < 0 is above the visible matrix

# SHAPES[piece][rotation] = (row, col) cells inside the SRS box. 0 = spawn, 1 = R (CW), 2 = 180, 3 = L.
SHAPES = {
    "T": [[(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (1, 2), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (2, 1)], [(0, 1), (1, 0), (1, 1), (2, 1)]],
    "S": [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 1), (1, 1), (1, 2), (2, 2)],
          [(1, 1), (1, 2), (2, 0), (2, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],
    "Z": [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 2), (1, 1), (1, 2), (2, 1)],
          [(1, 0), (1, 1), (2, 1), (2, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],
    "J": [[(0, 0), (1, 0), (1, 1), (1, 2)], [(0, 1), (0, 2), (1, 1), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 0), (2, 1)]],
    "L": [[(0, 2), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)],
          [(1, 0), (1, 1), (1, 2), (2, 0)], [(0, 0), (0, 1), (1, 1), (2, 1)]],
    "I": [[(1, 0), (1, 1), (1, 2), (1, 3)], [(0, 2), (1, 2), (2, 2), (3, 2)],
          [(2, 0), (2, 1), (2, 2), (2, 3)], [(0, 1), (1, 1), (2, 1), (3, 1)]],
    "O": [[(0, 1), (0, 2), (1, 1), (1, 2)]] * 4,
}

# ---- Keys -----------------------------------------------------------------
HOLD_KEY = "c"
KEY_NAMES = {"L": "ArrowLeft", "R": "ArrowRight", "CW": "ArrowUp", "CCW": "z", "SD": "ArrowDown"}
HARD_DROP = "Space"

# ---- SRS kicks (x right, y DOWN; converted from the usual y-up tables) -----
def _flip(t):
    return [(x, -y) for x, y in t]

_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
# Standard SRS I kicks. TETR.IO uses SRS+, whose I kicks differ slightly; a wrong
# guess only causes a rare misplacement, which the bot's board re-sync corrects.
_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}
KICKS = {p: {k: _flip(v) for k, v in (_I if p == "I" else _JLSTZ).items()} for p in SHAPES}

# ---- Precomputed row masks: PIECE_ROWS[piece][rot][x + 3] = ((dy, mask), ...) or None
def _build_rows():
    out = {}
    for p, rots in SHAPES.items():
        out[p] = []
        for cells in rots:
            per_x = []
            for x in range(-3, COLS):
                if any(not 0 <= x + c < COLS for _, c in cells):
                    per_x.append(None)
                    continue
                rows = {}
                for r, c in cells:
                    rows[r] = rows.get(r, 0) | (1 << (x + c))
                per_x.append(tuple(sorted(rows.items())))
            out[p].append(per_x)
    return out

PIECE_ROWS = _build_rows()


def to_bits(grid):
    """0/1 grid (list of lists, row 0 top) -> list of row bitmasks."""
    return [sum(1 << c for c, v in enumerate(row) if v) for row in grid]


def to_grid(board):
    return [[(row >> c) & 1 for c in range(COLS)] for row in board]


def _rows(piece, rot, x):
    if not -3 <= x < COLS:
        return None
    return PIECE_ROWS[piece][rot][x + 3]


def collides(board, rows, y):
    if rows is None:
        return True
    for dy, m in rows:
        yy = y + dy
        if yy >= ROWS:
            return True
        if yy >= 0 and board[yy] & m:
            return True
    return False


def _filled(board, y, x):
    if x < 0 or x >= COLS or y >= ROWS:
        return True
    return y >= 0 and (board[y] >> x) & 1


_FRONT = {0: ((0, 0), (0, 2)), 1: ((0, 2), (2, 2)), 2: ((2, 0), (2, 2)), 3: ((0, 0), (2, 0))}


def _tspin(board, rot, x, y, last):
    """0 = none, 1 = mini, 2 = full. `last` 1 = last input was a rotation, 2 = with the 5th kick."""
    if not last:
        return 0
    corners = sum(_filled(board, y + r, x + c) for r, c in ((0, 0), (0, 2), (2, 0), (2, 2)))
    if corners < 3:
        return 0
    if last == 2 or all(_filled(board, y + r, x + c) for r, c in _FRONT[rot]):
        return 2
    return 1


# ---- Movement search --------------------------------------------------------
@dataclass
class Placement:
    piece: str
    rot: int
    x: int
    y: int
    spin: int
    state: tuple = field(repr=False)
    parents: dict = field(repr=False, default=None)

    def keys(self):
        """Key presses from the start position, ending with a hard drop."""
        moves, s = [], self.state
        while self.parents[s] is not None:
            s, mv = self.parents[s]
            moves.append(mv)
        moves.reverse()
        # A soft drop right before the hard drop is useless: drop it.
        while moves and moves[-1] == "SD":
            moves.pop()
        return [KEY_NAMES[m] for m in moves] + [HARD_DROP]


_PL_CACHE = {}
_PL_CACHE_MAX = 30000


def placements(board, piece, start_x=SPAWN_X, start_y=SPAWN_Y, start_rot=0):
    """Every distinct lock position reachable from the start state, shortest input first.

    Spawn-start results are cached by (board, piece): the next turn's search revisits most of
    the boards this turn's search already expanded. Placements are read-only, so sharing is safe.
    """
    if start_x == SPAWN_X and start_y == SPAWN_Y and start_rot == 0:
        ck = (tuple(board), piece)
        hit = _PL_CACHE.get(ck)
        if hit is None:
            if len(_PL_CACHE) >= _PL_CACHE_MAX:
                _PL_CACHE.clear()
            hit = _PL_CACHE[ck] = _placements(board, piece, start_x, start_y, start_rot)
        return hit
    return _placements(board, piece, start_x, start_y, start_rot)


def _placements(board, piece, start_x, start_y, start_rot):
    rows0 = _rows(piece, start_rot, start_x)
    if collides(board, rows0, start_y):
        return []
    can_spin = piece == "T"
    rotates = piece != "O"
    kicks = KICKS[piece]
    start = (start_rot, start_x, start_y, 0)
    parents = {start: None}
    q = deque([start])
    found = {}
    out = []
    while q:
        s = q.popleft()
        r, x, y, last = s
        rows = _rows(piece, r, x)
        if collides(board, rows, y + 1):  # grounded: a lock candidate
            spin = _tspin(board, r, x, y, last) if can_spin else 0
            cells = tuple((y + dy, m) for dy, m in rows)
            key = (cells, spin)
            if key not in found:
                found[key] = True
                out.append(Placement(piece, r, x, y, spin, s, parents))
        nxt = []
        for mv, dx in (("L", -1), ("R", 1)):
            if not collides(board, _rows(piece, r, x + dx), y):
                nxt.append(((r, x + dx, y, 0), mv))
        ny = y
        while not collides(board, rows, ny + 1):
            ny += 1
        if ny != y:
            nxt.append(((r, x, ny, 0), "SD"))
        if rotates:
            for mv, nr in (("CW", (r + 1) % 4), ("CCW", (r + 3) % 4)):
                for i, (kx, ky) in enumerate(kicks[(r, nr)]):
                    if not collides(board, _rows(piece, nr, x + kx), y + ky):
                        flag = (2 if i == 4 else 1) if can_spin else 0
                        nxt.append(((nr, x + kx, y + ky, flag), mv))
                        break
        for ns, mv in nxt:
            if ns not in parents:
                parents[ns] = (s, mv)
                q.append(ns)
    return out


def lock(board, piece, rot, x, y):
    """Place and clear lines. Returns (new_board, lines) or (None, 0) on lock-out."""
    new = list(board)
    for dy, m in _rows(piece, rot, x):
        yy = y + dy
        if yy < 0:
            return None, 0
        new[yy] |= m
    kept = [row for row in new if row != FULL]
    lines = ROWS - len(kept)
    return [0] * lines + kept, lines


# ---- Attack (TETR.IO-like; adjust to your room's settings) ------------------
COMBO_TABLE = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5]
PC_ATTACK = 10


def attack(lines, spin, b2b, combo, pc):
    """Returns (garbage_sent, new_b2b, new_combo). combo = -1 means no active combo."""
    if lines == 0:
        return 0, b2b, -1
    combo += 1
    if spin == 2:
        base = (0, 2, 4, 6)[lines]
    elif spin == 1:
        base = (0, 0, 1, 2)[lines]
    else:
        base = (0, 0, 1, 2, 4)[lines]
    hard = lines == 4 or spin > 0
    if hard and b2b:
        base += 1
    base += COMBO_TABLE[min(combo, len(COMBO_TABLE) - 1)]
    if pc:
        base += PC_ATTACK
    return base, hard, combo


# ---- Evaluation -------------------------------------------------------------
# Feature set and starting weights follow Cold Clear's defaults (MinusKelvin), simplified.
W = dict(
    height=-39, top_half=-150, top_quarter=-511,
    bump=-24, bump_sq=-7,
    holes=-173, holes_sq=-3, hole_rows=-40,
    well_depth=57, max_well=4,
    tslot=300, tslot_partial=120,
    b2b_active=52,
    # per placement
    clear1=-143, clear2=-100, clear3=-58, clear4=390,
    tspin1=121, tspin2=410, tspin3=602, mini1=-158, mini2=-93,
    perfect=1500, b2b_clear=104, combo=150, wasted_t=-152,
    cancel=60, danger_clear=120,
)


def _tops(board):
    """Row index of the highest filled cell in each column (ROWS if the column is empty)."""
    tops = [ROWS] * COLS
    left = FULL
    for y in range(ROWS):
        row = board[y] & left
        if row:
            left &= ~row
            while row:
                low = row & -row
                tops[low.bit_length() - 1] = y
                row ^= low
            if not left:
                break
    return tops


def _tslots(board, tops=None):
    """Best T-spin-double slot quality on the board: 1.0 = ready, 0..0.6 = being built, 0 = none.

    A TSD slot's entry column is empty down to the slot's bottom row and filled there, so for
    each (centre, entry side) only one row y can work: y = top of the entry column - 2.
    That checks 16 candidates instead of scanning all 18 x 8.
    """
    if tops is None:
        tops = _tops(board)
    best = 0.0
    for cx in range(1, COLS - 1):
        b = 1 << cx
        m3 = 0b111 << (cx - 1)
        for entry in (cx - 1, cx + 1):
            y = tops[entry] - 2
            if y < 0 or y > ROWS - 3 or tops[cx] <= y + 2:
                continue
            r0, r1, r2 = board[y], board[y + 1], board[y + 2]
            if r1 & m3 or not (r2 >> (cx - 1)) & 1 or not (r2 >> (cx + 1)) & 1:
                continue
            over = cx + 1 if entry == cx - 1 else cx - 1
            if not (r0 >> over) & 1 or (r0 >> entry) & 1:
                continue
            if (r1 | m3) == FULL and (r2 | b) == FULL:
                return 1.0
            best = max(best, 0.6 * (bin(r1).count("1") + bin(r2).count("1")) / 16)
    return best


def features(board):
    heights = [0] * COLS
    seen = holes = hole_rows = 0
    for y in range(ROWS):
        row = board[y]
        h = ~row & seen & FULL
        if h:
            holes += bin(h).count("1")
            hole_rows += 1
        new = row & ~seen
        if new:
            for c in range(COLS):
                if (new >> c) & 1:
                    heights[c] = ROWS - y
            seen |= row
    well = min(range(COLS), key=heights.__getitem__)
    hs = [h for i, h in enumerate(heights) if i != well]
    bump = bump_sq = 0
    for a, b in zip(hs, hs[1:]):
        d = abs(a - b)
        bump += d
        bump_sq += d * d
    nb = [heights[i] for i in (well - 1, well + 1) if 0 <= i < COLS]
    depth = max(0, min(nb) - heights[well]) if nb else 0
    return heights, holes, hole_rows, bump, bump_sq, depth


def evaluate(board, b2b):
    heights, holes, hole_rows, bump, bump_sq, depth = features(board)
    mh = max(heights)
    ts = _tslots(board, [ROWS - h for h in heights])
    if ts >= 1.0:
        holes = max(0, holes - 1)  # the overhang of a T-slot is not a real hole
    s = (W["height"] * mh + W["top_half"] * max(0, mh - 10) + W["top_quarter"] * max(0, mh - 15)
         + W["bump"] * bump + W["bump_sq"] * bump_sq
         + W["holes"] * holes + W["holes_sq"] * holes * holes + W["hole_rows"] * hole_rows
         + W["well_depth"] * min(depth, W["max_well"])
         + (W["tslot"] if ts >= 1.0 else W["tslot_partial"] * ts))
    if b2b:
        s += W["b2b_active"]
    return s, mh


def step_reward(piece, lines, spin, sent, b2b_before, combo, pc, danger, incoming):
    r = 0.0
    if lines:
        if spin == 2:
            r += (0, W["tspin1"], W["tspin2"], W["tspin3"])[lines]
        elif spin == 1:
            r += (0, W["mini1"], W["mini2"], 0)[lines]
        else:
            r += (0, W["clear1"], W["clear2"], W["clear3"], W["clear4"])[lines] * (1 - danger)
        if b2b_before and (lines == 4 or spin):
            r += W["b2b_clear"]
        r += W["combo"] * COMBO_TABLE[min(combo, len(COMBO_TABLE) - 1)]
        r += W["cancel"] * min(sent, incoming)
        r += W["danger_clear"] * lines * danger
        if pc:
            r += W["perfect"]
    if piece == "T" and not spin:
        r += W["wasted_t"]
    return r


# ---- Beam search ------------------------------------------------------------
@dataclass
class Decision:
    use_hold: bool
    placement: Placement
    score: float


def _options(pieces, idx, hold, can_hold=True):
    """(piece_to_play, new_hold, new_idx, used_hold) for the node's turn."""
    if idx >= len(pieces):
        return []
    opts = [(pieces[idx], hold, idx + 1, False)]
    if can_hold:
        if hold is not None:
            if hold != pieces[idx]:
                opts.append((hold, pieces[idx], idx + 1, True))
        elif idx + 1 < len(pieces):
            opts.append((pieces[idx + 1], pieces[idx], idx + 2, True))
    return opts


def search(board, current, queue, hold=None, b2b=False, combo=-1, incoming=0,
           depth=3, beam=12, start_x=SPAWN_X, allow_hold=True, start_y=SPAWN_Y, start_rot=0):
    """Pick the best move for `current`, looking `depth` placements ahead through the queue."""
    pieces = [current] + list(queue)
    _, mh0 = evaluate(board, b2b)
    # (acc_reward, board, hold, idx, b2b, combo, first_decision)
    frontier = [(0.0, board, hold, 0, b2b, combo, None)]
    best_leaf = None
    for d in range(depth):
        children = []
        for acc, bd, hd, idx, bb, cb, first in frontier:
            for piece, nh, nidx, used in _options(pieces, idx, hd, allow_hold or d > 0):
                if d == 0 and not used:
                    plist = placements(bd, piece, start_x, start_y, start_rot)
                else:
                    plist = placements(bd, piece)
                for pl in plist:
                    new, lines = lock(bd, piece, pl.rot, pl.x, pl.y)
                    if new is None:
                        continue
                    pc = lines > 0 and not any(new)
                    sent, nb, nc = attack(lines, pl.spin, bb, cb, pc)
                    ev, mh = evaluate(new, nb)
                    danger = min(1.0, max(0.0, (mh + incoming - 8) / 8))
                    rew = acc + step_reward(piece, lines, pl.spin, sent, bb, nc, pc, danger, incoming)
                    dec = first or Decision(used, pl, 0.0)
                    children.append((rew + ev, rew, new, nh, nidx, nb, nc, dec))
        if not children:
            break
        children.sort(key=lambda t: t[0], reverse=True)
        # Different move orders often reach the same position: keep only the best copy,
        # so the beam holds `beam` DIFFERENT positions instead of duplicates.
        uniq, seen = [], set()
        for c in children:
            k = (tuple(c[2]), c[3], c[4], c[5])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
                if len(uniq) == beam:
                    break
        children = uniq
        best_leaf = children[0]
        frontier = [(c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in children]
    if best_leaf is None:
        return None
    dec = best_leaf[7]
    dec.score = best_leaf[0]
    return dec


# ---- Piece identification (shape-based, skin independent) ------------------
def _norm(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells), mc


_SPAWN_LOOKUP = {}
for _p, _rots in SHAPES.items():
    _n, _off = _norm(_rots[0])
    _SPAWN_LOOKUP[_n] = (_p, _off)


def identify_piece(cells):
    """(piece, box_x) for 4 cells in spawn orientation, else None."""
    if len(cells) != 4:
        return None
    n, mc = _norm(cells)
    hit = _SPAWN_LOOKUP.get(n)
    return None if hit is None else (hit[0], mc - hit[1])


def identify_shape(cells):
    """Piece name from any 4-cell spawn-orientation pattern (used for NEXT / HOLD)."""
    hit = identify_piece(list(cells))
    return hit[0] if hit else None
