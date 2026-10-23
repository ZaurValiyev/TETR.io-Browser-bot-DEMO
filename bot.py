"""Educational TETR.IO versus bot: Playwright (input) + OpenCV (vision) + search AI.

Usage:
    python bot.py calibrate        # drag a box around the 10x20 matrix
    python bot.py debug            # show what the bot reads (board, NEXT, HOLD, garbage) + speed
    python bot.py image shot.png   # same as debug, but on a saved screenshot (no browser)
    python bot.py keytest          # check that key presses reach the game (run in Zen)
    python bot.py play             # play (Ctrl+C in this terminal to stop)
    python bot.py play nohold      # play without hold

Required TETR.IO settings (Config > Handling / Controls):
    SDF = infinite (one Down press drops instantly), default keys:
    Left/Right, Up = rotate CW, Z = rotate CCW, C = hold, Space = hard drop.
    Recommended: ghost piece off or very transparent.

Use only in Zen / solo / PRIVATE custom rooms where your opponent agreed to play a bot.
TETR.IO forbids bots in ranked, Quick Play, public lobbies and leaderboards.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

import tetris_ai as ai
from tetris_ai import COLS, HOLD_KEY, ROWS, SHAPES, SPAWN_X, identify_piece, identify_shape

CONFIG = Path("config.json")
PROFILE_DIR = Path("browser_profile")
VIEWPORT = {"width": 1280, "height": 720}

# ---- Tunables -------------------------------------------------------------
# Search
DEPTH = 3                # placements looked ahead (current + 2 from the queue)
BEAM = 12                # positions kept per level (higher = stronger, slower)
# Timing
KEY_HOLD = 0.020         # how long each key stays DOWN (>= 1 frame, or TETR.IO may miss it)
KEY_DELAY = 0.012        # pause after releasing a key
HOLD_DELAY = 0.03        # after pressing hold, let the swapped piece spawn
MAX_STALE_FRAMES = 12
SETTLE_FRAMES = 6        # extra frames to wait out animations before believing a mismatch
SETTLE_WAIT = 0.035      # seconds between those frames
VERIFY = "auto"          # True = check every piece before hard dropping, False = never,
                         # "auto" = only risky moves (soft drop, spin, many keys) and the first
                         # VERIFY_WARMUP pieces; switches to always if a correction is ever needed
VERIFY_WARMUP = 15
VERIFY_MIN_KEYS = 4      # in "auto", verify moves with at least this many key presses
PIPELINE = True          # think about the next piece while this one's keys are being pressed
VERIFY_WAIT = 0.03       # let the game draw the last move before checking
VERIFY_TRIES = 2         # max correction rounds per piece
CLEAR_ANIM = 0.25        # after a line clear, ignore "stable" screens younger than this
# Board vision
BUFFER_ROWS = 2          # rows read ABOVE the matrix (pieces spawn there)
TOP_MASK_ROWS = 3        # top matrix rows treated as "falling piece" area
SAT_MIN, VAL_MIN = 70, 90
BUFFER_BG_DIST = 60
GRAY_V_MIN, GRAY_V_MAX = 55, 200
GHOST_FILTER = True      # remove a ghost piece that was read as stack
# Panels, in board-cell units relative to the matrix (x = left edge, top = top edge).
# Run `python bot.py debug` and check the drawn boxes if your layout differs.
NEXT_BOX = (0.3, 4.9, 1.3, 16.5)     # x from right edge of matrix: (x0, x1), y from top: (y0, y1)
HOLD_BOX = (-5.6, -0.9, 1.1, 3.9)    # x from left edge of matrix
PANEL_V_MIN = 70                     # brighter than this = part of a piece (also grey "used" hold)
GARBAGE_X = -0.4                     # garbage meter column, cells left of the matrix
CAPTURE_FORMAT = "jpeg"   # much faster than png; reads identically
JPEG_QUALITY = 85


# ---- Capture ----------------------------------------------------------------
def _geom(roi):
    x, y, w, h = roi
    cell = w / COLS
    bottom = y + h
    return x, w, cell, bottom, bottom - ROWS * cell


class Capture:
    """Grabs only the area around the board (board, HOLD, NEXT, garbage meter) via CDP."""

    def __init__(self, page, roi):
        self.page = page
        x, w, cell, bottom, top = _geom(roi)
        x0 = max(0, int(x + HOLD_BOX[0] * cell - 4))
        x1 = min(VIEWPORT["width"], int(x + w + NEXT_BOX[1] * cell + 4))
        y0 = max(0, int(top - (BUFFER_ROWS + 1) * cell))
        y1 = min(VIEWPORT["height"], int(bottom + cell / 2))
        self.clip = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "scale": 1}
        self.roi = (roi[0] - x0, roi[1] - y0, roi[2], roi[3])  # ROI in the cropped image
        self.fast = True
        try:
            self.cdp = page.context.new_cdp_session(page)
        except Exception:
            self.cdp = None

    def grab(self):
        if self.cdp is not None:
            args = {"format": CAPTURE_FORMAT, "clip": self.clip}
            if CAPTURE_FORMAT == "jpeg":
                args["quality"] = JPEG_QUALITY
            if self.fast:
                args["optimizeForSpeed"] = True
            try:
                data = base64.b64decode(self.cdp.send("Page.captureScreenshot", args)["data"])
            except Exception:
                if not self.fast:
                    raise
                self.fast = False  # older Chromium: retry without optimizeForSpeed
                return self.grab()
        else:
            c = self.clip
            data = self.page.screenshot(clip={k: c[k] for k in ("x", "y", "width", "height")})
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---- Vision: board ------------------------------------------------------------
_PATCH = np.arange(-2, 3)


def _patch_median(arr, ys, xs):
    """Median colour of the 5x5 patch around every (ys[i], xs[j]) -> (len(ys), len(xs), 3)."""
    Y = ys[:, None, None, None] + _PATCH[None, None, :, None]
    X = xs[None, :, None, None] + _PATCH[None, None, None, :]
    return np.median(arr[Y, X].reshape(len(ys), len(xs), 25, 3), axis=2)


def read_grid(img, roi):
    """0/1 grid with BUFFER_ROWS extra rows on top; index BUFFER_ROWS = top matrix row.
    One numpy pass over all cells (same result as reading them one by one, ~4x faster)."""
    x, w, cell, bottom, _ = _geom(roi)
    total = BUFFER_ROWS + ROWS
    H, Wd = img.shape[:2]
    cy = (bottom - (total - np.arange(total) - 0.5) * cell).astype(int)
    valid = (cy >= 3) & (cy <= H - 4)
    cy = np.clip(cy, 3, H - 4)
    cx = (x + (np.arange(COLS) + 0.5) * cell).astype(int)
    med = _patch_median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cy, cx)
    s, v = med[..., 1], med[..., 2]
    filled = ((s > SAT_MIN) & (v > VAL_MIN)) | ((s <= SAT_MIN) & (v > GRAY_V_MIN) & (v < GRAY_V_MAX))
    filled &= valid[:, None]
    # Above the matrix the background can be colourful: drop cells that look like it.
    rxs = np.array([rx for rx in (int(x - 12), int(x + w + 12)) if 3 <= rx < Wd - 3])
    if len(rxs) and BUFFER_ROWS:
        ys = cy[:BUFFER_ROWS]
        px = _patch_median(img, ys, cx)
        ref = _patch_median(img, ys, rxs)
        dist = np.linalg.norm(px[:, :, None, :] - ref[:, None, :, :], axis=3).min(axis=2)
        filled[:BUFFER_ROWS] &= dist >= BUFFER_BG_DIST
    return filled.astype(int).tolist()


def spawn_cells(grid):
    last = BUFFER_ROWS + TOP_MASK_ROWS
    return [(r, c) for r in range(last) for c in range(COLS) if grid[r][c]]


def detect_piece(grid):
    """Falling piece by shape: (piece, box_x) or None."""
    last = BUFFER_ROWS + TOP_MASK_ROWS
    filled = {(r, c) for r in range(last) for c in range(COLS) if grid[r][c]}
    in_buffer = {p for p in filled if p[0] < BUFFER_ROWS}
    if not in_buffer:
        return identify_piece(spawn_cells(grid))
    best = None
    for piece, rots in SHAPES.items():
        for top in range(0, BUFFER_ROWS + 1):
            for bx in range(-3, COLS):
                cells = {(top + r, bx + c) for r, c in rots[0]}
                if not cells <= filled or not any(r < BUFFER_ROWS for r, _ in cells):
                    continue
                key = (len(in_buffer - cells), abs(bx - SPAWN_X))
                if best is None or key < best[0]:
                    best = (key, piece, bx)
    return (best[1], best[2]) if best else identify_piece(spawn_cells(grid))


def find_floating(grid):
    """A falling piece that has already dropped below the spawn area.

    Looks for a group of exactly 4 touching cells in spawn orientation with
    nothing under it (stack pieces always rest on something).
    Returns (piece, box_x, cells) with cells as (grid_row, col), or None.
    """
    total = len(grid)
    seen = set()
    for r0 in range(BUFFER_ROWS, total):
        for c0 in range(COLS):
            if not grid[r0][c0] or (r0, c0) in seen:
                continue
            comp, stack = [], [(r0, c0)]
            seen.add((r0, c0))
            while stack and len(comp) <= 4:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < total and 0 <= nc < COLS and grid[nr][nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            if len(comp) != 4:
                continue
            cells = set(comp)
            if any(r + 1 >= total or (grid[r + 1][c] and (r + 1, c) not in cells) for r, c in cells):
                continue  # resting on the floor or the stack: not falling
            hit = identify_piece(comp)
            if hit:
                return hit[0], hit[1], comp
    return None


def board_bits(grid, remove=()):
    """The 20 matrix rows as bitmasks, falling-piece area and `remove` cells cleared."""
    rows = [row[:] for row in grid[BUFFER_ROWS:]]
    for r in range(TOP_MASK_ROWS):
        rows[r] = [0] * COLS
    for r, c in remove:
        if r >= BUFFER_ROWS:
            rows[r - BUFFER_ROWS][c] = 0
    return ai.to_bits(rows)


def strip_ghost(board, piece, x):
    """If the ghost of `piece` (spawn rotation at column x) was read as stack, remove it."""
    rows = ai._rows(piece, 0, x)
    if rows is None:
        return board
    for y in range(TOP_MASK_ROWS, ROWS):
        if any(y + dy >= ROWS or board[y + dy] & m != m for dy, m in rows):
            continue
        b2 = list(board)
        for dy, m in rows:
            b2[y + dy] &= ~m
        yy = ai.SPAWN_Y
        if ai.collides(b2, rows, yy):
            continue
        while not ai.collides(b2, rows, yy + 1):
            yy += 1
        if yy == y:
            return b2
    return board


def landing_cells(board, piece, x):
    """Bitmask rows where `piece` (spawn rotation, column x) would land: its ghost position."""
    rows = ai._rows(piece, 0, x)
    if rows is None or ai.collides(board, rows, ai.SPAWN_Y):
        return None
    y = ai.SPAWN_Y
    while not ai.collides(board, rows, y + 1):
        y += 1
    return [(y + dy, m) for dy, m in rows]


def clear_cells(board, cells):
    b = list(board)
    for y, m in cells or ():
        if 0 <= y < ROWS:
            b[y] &= ~m
    return b


def _match_piece(piece, cells):
    """Which (rot, x, y) of `piece` covers exactly these cells, if any."""
    want = set(cells)
    for rot in range(4):
        for x in range(-3, COLS):
            for y in range(-3, ROWS):
                if {(y + r, x + c) for r, c in SHAPES[piece][rot]} == want:
                    return rot, x, y
    return None


def explain_resync(prev, vis, pl):
    """Compare where the last piece was planned to land with what the screen shows."""
    if prev is None or pl is None:
        return "no previous plan to compare with"
    new = sorted((y, c) for y in range(TOP_MASK_ROWS, ROWS) for c in range(COLS)
                 if (vis[y] >> c) & 1 and not (prev[y] >> c) & 1)
    gone = sum(1 for y in range(TOP_MASK_ROWS, ROWS) for c in range(COLS)
               if (prev[y] >> c) & 1 and not (vis[y] >> c) & 1)
    plan = f"planned {pl.piece} rot={pl.rot} x={pl.x}"
    if gone:
        return f"{plan}; {gone} old cells vanished (line clear or misread), can't compare"
    if len(new) != 4:
        return f"{plan}; screen shows {len(new)} new cells instead of 4 -> vision noise / ghost / missed lock"
    got = _match_piece(pl.piece, new)
    if got is None:
        return f"{plan}; new cells {new} are not a {pl.piece} -> wrong piece played (hold or detection)"
    rot, x, _ = got
    if (rot, x) == (pl.rot, pl.x):
        return f"{plan}; landed in the right column/rotation but at another height -> soft drop (check SDF = inf)"
    hint = []
    if rot != pl.rot and not (pl.piece in "SZI" and rot % 2 == pl.rot % 2):
        hint.append("rotation keys missed or wrong bindings")
    if x != pl.x:
        wall = x in (min(xx for xx in range(-3, COLS) if ai._rows(pl.piece, rot, xx)),
                     max(xx for xx in range(-3, COLS) if ai._rows(pl.piece, rot, xx)))
        hint.append("piece slid to the wall: DAS shorter than KEY_HOLD" if wall and abs(x - pl.x) > 1
                    else f"moved {x - SPAWN_X:+d} columns instead of {pl.x - SPAWN_X:+d}: taps missed")
    return f"{plan}; actually rot={rot} x={x} -> " + ", ".join(hint or ["unclear"])


def find_active(grid, board, piece):
    """Pose (rot, x, y) of the falling piece: the highest place its shape fits in the cells
    that are on screen but not in our board (the ghost, if visible, is the lower copy)."""
    cand = set()
    for r, row in enumerate(grid):
        y = r - BUFFER_ROWS
        for c, v in enumerate(row):
            if v and (y < 0 or not (board[y] >> c) & 1):
                cand.add((y, c))
    if len(cand) < 4:
        return None
    for y in range(-BUFFER_ROWS - 1, ROWS):
        for rot in range(4 if piece != "O" else 1):
            for x in range(-3, COLS):
                cells = {(y + r, x + c) for r, c in SHAPES[piece][rot]}
                if cells <= cand and not ai.collides(board, ai._rows(piece, rot, x), y):
                    return rot, x, y
    return None


def drop_cells(board, piece, rot, x, y):
    rows = ai._rows(piece, rot, x)
    while not ai.collides(board, rows, y + 1):
        y += 1
    return tuple((y + dy, m) for dy, m in rows)


def correction(board, pl, pose):
    """Keys (without the hard drop) to reach pl's landing spot from the real pose,
    or the best other spot if that one can't be reached. Returns (placement, keys)."""
    rot, x, y = pose
    target = drop_cells(board, pl.piece, pl.rot, pl.x, pl.y)
    options = ai.placements(board, pl.piece, x, y, rot)
    for p in options:
        if drop_cells(board, p.piece, p.rot, p.x, p.y) == target and p.spin == pl.spin:
            return p, p.keys()[:-1]
    for p in options:
        if drop_cells(board, p.piece, p.rot, p.x, p.y) == target:
            return p, p.keys()[:-1]
    dec = ai.search(board, pl.piece, [], depth=1, beam=1, allow_hold=False,
                    start_x=x, start_y=y, start_rot=rot)
    return (dec.placement, dec.placement.keys()[:-1]) if dec else (None, None)


# ---- Vision: panels -----------------------------------------------------------
def _box(roi, spec, right):
    x, w, cell, _, top = _geom(roi)
    ox = x + w if right else x
    return (int(ox + spec[0] * cell), int(ox + spec[1] * cell),
            int(top + spec[2] * cell), int(top + spec[3] * cell))


def read_panel(img, roi, spec, right):
    """Pieces shown in a panel (NEXT or HOLD), top to bottom, identified by shape."""
    _, _, cell, _, _ = _geom(roi)
    x0, x1, y0, y1 = _box(roi, spec, right)
    x0, y0 = max(0, x0), max(0, y0)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = ((v > PANEL_V_MIN) & ~((s < 40) & (v > 200))).astype(np.uint8)  # pieces, not white UI
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < 1.5 * cell * cell:
            continue
        gw, gh = round(bw / cell), round(bh / cell)
        if not (2 <= gw <= 4 and 1 <= gh <= 2):
            continue
        cells = []
        for r in range(gh):
            for c in range(gw):
                cy = int(by + (r + 0.5) * bh / gh)
                cx = int(bx + (c + 0.5) * bw / gw)
                if mask[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3].mean() > 0.5:
                    cells.append((r, c))
        name = identify_shape(cells)
        if name:
            found.append((by, name))
    return [name for _, name in sorted(found)]


def read_incoming(img, roi):
    """Rows of garbage waiting in the red meter left of the matrix (approximate)."""
    x, _, cell, bottom, top = _geom(roi)
    gx = int(x + GARBAGE_X * cell)
    if not 1 <= gx < img.shape[1] - 1:
        return 0
    col = cv2.cvtColor(img[int(top):int(bottom), gx - 1:gx + 2], cv2.COLOR_BGR2HSV).reshape(-1, 3, 3)
    h, s, v = col[..., 0], col[..., 1], col[..., 2]
    red = ((h < 12) | (h > 165)) & (s > 120) & (v > 120)
    return int(round(red.any(axis=1).sum() / cell))


def read_state(img, roi):
    grid = read_grid(img, roi)
    piece, falling = detect_piece(grid), ()
    if piece is None:  # it may already have fallen below the spawn area
        fl = find_floating(grid)
        if fl:
            piece, falling = (fl[0], fl[1]), fl[2]
    return {
        "grid": grid,
        "piece": piece,
        "falling": falling,
        "queue": read_panel(img, roi, NEXT_BOX, right=True),
        "hold": (read_panel(img, roi, HOLD_BOX, right=False) or [None])[0],
        "incoming": read_incoming(img, roi),
    }


def annotate(img, roi, grid):
    out = img.copy()
    x, y, w, h = roi
    _, _, cell, bottom, _ = _geom(roi)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
    total = BUFFER_ROWS + ROWS
    for i in range(total):
        for c in range(COLS):
            p = (int(x + (c + 0.5) * cell), int(bottom - (total - i - 0.5) * cell))
            cv2.circle(out, p, 2, (0, 0, 255) if grid[i][c] else (255, 255, 0), -1)
    for spec, right in ((NEXT_BOX, True), (HOLD_BOX, False)):
        x0, x1, y0, y1 = _box(roi, spec, right)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1)
    gx = int(x + GARBAGE_X * cell)
    cv2.line(out, (gx, int(bottom - ROWS * cell)), (gx, int(bottom)), (0, 165, 255), 1)
    return out


def describe(state):
    grid = state["grid"]
    low = "  (already fallen below spawn)" if state.get("falling") else ""
    lines = [f"piece: {state['piece']}{low}  queue: {state['queue']}  hold: {state['hold']}  "
             f"incoming: {state['incoming']}   (~ above matrix, ^ spawn area)"]
    for i, row in enumerate(grid):
        tag = "~" if i < BUFFER_ROWS else ("^" if i < BUFFER_ROWS + TOP_MASK_ROWS else " ")
        lines.append(tag + "".join("#" if v else "." for v in row))
    return "\n".join(lines)


def suggest(state):
    """Run the AI on a captured state and print its choice."""
    if not state["piece"]:
        print("no falling piece detected: nothing to decide")
        return
    piece, x = state["piece"]
    board = board_bits(state["grid"], state["falling"])
    if GHOST_FILTER:
        board = strip_ghost(board, piece, x)
    t0 = time.perf_counter()
    dec = ai.search(board, piece, state["queue"][:5], state["hold"], incoming=state["incoming"],
                    depth=DEPTH, beam=BEAM, start_x=x)
    dt = (time.perf_counter() - t0) * 1000
    if dec:
        pl = dec.placement
        print(f"AI ({dt:.0f} ms): {'HOLD, then ' if dec.use_hold else ''}{pl.piece} rot={pl.rot} "
              f"x={pl.x} spin={pl.spin} keys={pl.keys()}")


# ---- Browser --------------------------------------------------------------
ADBLOCK_DIR = Path("ublock")


def open_game(p):
    args = []
    if (ADBLOCK_DIR / "manifest.json").exists():
        ext = str(ADBLOCK_DIR.resolve())
        args = [f"--disable-extensions-except={ext}", f"--load-extension={ext}"]
    ctx = p.chromium.launch_persistent_context(
        str(PROFILE_DIR), headless=False, viewport=VIEWPORT, device_scale_factor=1, args=args)
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://tetr.io/")
    return ctx, page


def load_roi():
    if not CONFIG.exists():
        sys.exit("No config.json yet: run `python bot.py calibrate` first.")
    roi = tuple(json.loads(CONFIG.read_text())["roi"])
    if not 0.40 <= roi[2] / roi[3] <= 0.55:
        sys.exit(f"Saved box {roi[2]}x{roi[3]} is not board-shaped: run `python bot.py calibrate` "
                 "and box ONLY the 10x20 matrix (not HOLD / NEXT).")
    return roi


def tap(page, key):
    """Press a key long enough for the game to see it (instant press/release can be missed)."""
    page.keyboard.down(key)
    time.sleep(KEY_HOLD)
    page.keyboard.up(key)
    time.sleep(KEY_DELAY)


def calibrate():
    with sync_playwright() as p:
        ctx, page = open_game(p)
        input("Start a game so the matrix is visible, then press Enter here...")
        img = cv2.imdecode(np.frombuffer(page.screenshot(), np.uint8), cv2.IMREAD_COLOR)
        print("Drag a box tightly around the 10x20 matrix, press ENTER/SPACE to confirm.")
        x, y, w, h = cv2.selectROI("calibrate", img, showCrosshair=False)
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            sys.exit("No selection made.")
        CONFIG.write_text(json.dumps({"roi": [int(x), int(y), int(w), int(h)]}))
        print("Saved config.json", (x, y, w, h))
        ctx.close()


def debug():
    roi = load_roi()
    with sync_playwright() as p:
        ctx, page = open_game(p)
        input("Start a game, then press Enter to capture...")
        cap = Capture(page, roi)
        t0 = time.perf_counter()
        for _ in range(10):
            img = cap.grab()
        print(f"capture: {(time.perf_counter() - t0) * 100:.0f} ms per frame")
        state = read_state(img, cap.roi)
        print(describe(state))
        suggest(state)
        cv2.imwrite("debug.png", annotate(img, cap.roi, state["grid"]))
        print("Saved debug.png (green = board, magenta = NEXT/HOLD areas, orange = garbage meter)")
        ctx.close()


def keytest():
    """Check that key presses reach the game: move the piece and see if it really moved."""
    roi = load_roi()
    with sync_playwright() as p:
        ctx, page = open_game(p)
        input("Start a Zen game (don't touch the keyboard), then press Enter...")
        cap = Capture(page, roi)
        page.mouse.click(VIEWPORT["width"] // 2, VIEWPORT["height"] // 2)  # make sure the game has focus

        def piece_x():
            for _ in range(30):
                st = read_state(cap.grab(), cap.roi)
                if st["piece"]:
                    return st["piece"]
                time.sleep(0.02)
            return None

        ok = 0
        for direction, key in (("left", "ArrowLeft"), ("right", "ArrowRight")):
            start = piece_x()
            if start is None:
                print("No falling piece found: run `python bot.py debug` first.")
                break
            for _ in range(2):
                tap(page, key)
            time.sleep(0.05)
            end = piece_x()
            moved = None if end is None else end[1] - start[1]
            want = -2 if direction == "left" else 2
            print(f"2 x {key}: piece {start} -> {end}, moved {moved} (expected {want})")
            ok += moved == want
        print("Keys OK." if ok == 2 else
              "Keys are being missed: raise KEY_HOLD / KEY_DELAY in bot.py, "
              "and check the keys in TETR.IO Config > Controls.")
        ctx.close()


def debug_image(path):
    roi = load_roi()
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Can't read {path}")
    state = read_state(img, roi)
    print(describe(state))
    suggest(state)
    cv2.imwrite("debug_image.png", annotate(img, roi, state["grid"]))
    print("Saved debug_image.png")


def play(use_hold=True):
    roi = load_roi()
    with sync_playwright() as p:
        ctx, page = open_game(p)
        input("Start a Zen game or a private room, wait for GO, then press Enter...")
        cap = Capture(page, roi)
        sim, b2b, combo, tracked_hold = None, False, -1, None
        last_queue, shift, stale = None, 0, 0
        placed, sent_total, t_first = 0, 0, None
        miss_since, saved = None, 0
        expected, prev_board, last_pl, resyncs, last_lines = None, None, None, 0, 0
        corrections = 0
        verify_all = VERIFY is True
        pool = ThreadPoolExecutor(max_workers=1) if PIPELINE else None
        pre = None   # (inputs it was computed for, future) of the next piece's decision
        reused = 0
        try:
            while True:
                img = cap.grab()
                st = read_state(img, cap.roi)
                if st["piece"] is None:
                    now = time.time()
                    miss_since = miss_since or now
                    if now - miss_since > 2.0 and saved < 3:
                        saved += 1
                        cv2.imwrite(f"stuck_{saved}.png", annotate(img, cap.roi, st["grid"]))
                        print(f"\nstuck: saved stuck_{saved}.png\n{describe(st)}")
                        miss_since = now
                    time.sleep(0.01)
                    continue
                miss_since = None
                queue = st["queue"][:5]
                # Wait until this frame really shows the NEW piece: the queue must have shifted.
                fresh = True
                if last_queue is not None:
                    want = last_queue[shift:]
                    if queue[:len(want)] != want:
                        if stale < MAX_STALE_FRAMES:
                            stale += 1
                            continue
                        fresh = False  # queue never matched: fall back to what the screen shows
                stale = 0
                piece, cur_x = st["piece"]
                # Once we're in sync we KNOW the next piece from the queue; the screen guess of the
                # falling piece is only trusted for the first piece or after losing track.
                if expected is not None and fresh:
                    piece, cur_x = expected, SPAWN_X
                vis = board_bits(st["grid"], st["falling"])
                if sim is not None:
                    # The ghost sits exactly where the piece would land: that spot is empty for sure.
                    vis_clean = clear_cells(vis, landing_cells(sim, piece, cur_x))
                else:
                    vis_clean = strip_ghost(vis, piece, cur_x) if GHOST_FILTER else vis
                same = lambda b: b[TOP_MASK_ROWS:] == sim[TOP_MASK_ROWS:]
                if sim is not None and not same(vis_clean):
                    # Don't trust one frame: line-clear and lock animations draw fading cells.
                    # Re-read until the screen matches our model or shows the same board twice.
                    last_read = vis_clean
                    t_settle = time.time()
                    min_wait = CLEAR_ANIM if last_lines else 0.0
                    for _ in range(SETTLE_FRAMES + (8 if last_lines else 0)):
                        time.sleep(SETTLE_WAIT)
                        img = cap.grab()
                        st2 = read_state(img, cap.roi)
                        v2 = board_bits(st2["grid"], st2["falling"])
                        v2 = clear_cells(v2, landing_cells(sim, piece, cur_x))
                        st["grid"] = st2["grid"]
                        if same(v2) or (v2 == last_read and time.time() - t_settle >= min_wait):
                            vis_clean = v2
                            break
                        last_read = v2
                if sim is not None and same(vis_clean):
                    board = sim
                else:
                    if sim is not None:
                        resyncs += 1
                        why = explain_resync(prev_board, vis_clean, last_pl)
                        print(f"\nresync #{resyncs}: {why}")
                        if resyncs <= 5:
                            cv2.imwrite(f"resync_{resyncs}.png", annotate(img, cap.roi, st["grid"]))
                    board = strip_ghost(vis_clean, piece, cur_x) if GHOST_FILTER else vis_clean
                hold = (st["hold"] or tracked_hold) if use_hold else None
                dec = None
                if pre is not None:
                    (p_board, p_piece, p_queue, p_hold, p_b2b, p_combo, p_inc), fut = pre
                    pre = None
                    if (cur_x == SPAWN_X and p_piece == piece and p_board == tuple(board)
                            and tuple(queue[:len(p_queue)]) == p_queue and p_hold == hold
                            and (p_b2b, p_combo, p_inc) == (b2b, combo, st["incoming"])):
                        dec = fut.result()  # already thought about while the last keys went out
                        reused += dec is not None
                if dec is None:
                    dec = ai.search(board, piece, queue, hold, b2b, combo, st["incoming"],
                                    depth=DEPTH, beam=BEAM, start_x=cur_x, allow_hold=use_hold)
                if dec is None:
                    print("\nno legal move (topped out?)")
                    sim, last_queue, expected, pre = None, None, None, None
                    time.sleep(0.2)
                    continue
                pl = dec.placement
                keys = pl.keys()
                if pool is not None:
                    # Predict the position after this move and start the next search now: it runs
                    # while the keys below are pressed (time.sleep releases the GIL).
                    p_new, p_lines = ai.lock(board, pl.piece, pl.rot, pl.x, pl.y)
                    p_shift = (2 if hold is None else 1) if dec.use_hold else 1
                    p_queue = tuple(queue[p_shift:])
                    if p_new is not None and len(queue) >= p_shift + 1 and len(p_queue) >= 3:
                        p_pc = p_lines > 0 and not any(p_new)
                        _, p_b2b, p_combo = ai.attack(p_lines, pl.spin, b2b, combo, p_pc)
                        p_hold = piece if dec.use_hold else hold
                        p_piece = queue[p_shift - 1]
                        inputs = (tuple(p_new), p_piece, p_queue, p_hold, p_b2b, p_combo, st["incoming"])
                        pre = (inputs, pool.submit(ai.search, p_new, p_piece, list(p_queue), p_hold,
                                                   p_b2b, p_combo, st["incoming"], depth=DEPTH,
                                                   beam=BEAM, allow_hold=use_hold))
                if dec.use_hold:
                    tap(page, HOLD_KEY)
                    time.sleep(HOLD_DELAY)
                for key in keys[:-1]:
                    tap(page, key)
                risky = (pl.spin or "ArrowDown" in keys or len(keys) - 1 >= VERIFY_MIN_KEYS
                         or placed < VERIFY_WARMUP)
                if VERIFY and (verify_all or risky):
                    for _ in range(VERIFY_TRIES):
                        time.sleep(VERIFY_WAIT)
                        g3 = read_grid(cap.grab(), cap.roi)
                        pose = find_active(g3, board, pl.piece)
                        if pose is None or (drop_cells(board, pl.piece, *pose)
                                            == drop_cells(board, pl.piece, pl.rot, pl.x, pl.y)):
                            break
                        new_pl, fix = correction(board, pl, pose)
                        if new_pl is None:
                            break
                        corrections += 1
                        if VERIFY == "auto":
                            verify_all = True  # keys get missed on this setup: check every piece
                        pl = new_pl
                        for key in fix:
                            tap(page, key)
                tap(page, keys[-1])
                # Update our model of the game.
                new, lines = ai.lock(board, pl.piece, pl.rot, pl.x, pl.y)
                pc = new is not None and lines > 0 and not any(new)
                sent, b2b, combo = ai.attack(lines, pl.spin, b2b, combo, pc)
                sent_total += sent
                if dec.use_hold:
                    shift = 2 if hold is None else 1  # hold from empty consumes one queue piece
                    tracked_hold = piece
                else:
                    shift = 1
                expected = queue[shift - 1] if len(queue) >= shift else None
                prev_board, last_pl, last_lines = board, pl, lines
                sim, last_queue = new, queue
                placed += 1
                t_first = t_first or time.time()
                pps = placed / max(time.time() - t_first, 1e-6) if placed > 1 else 0
                spin = {0: "", 1: " T-spin mini", 2: " T-SPIN"}[pl.spin]
                held = " (from hold)" if dec.use_hold else ""
                print(f"#{placed} {pl.piece}{held}{spin} lines={lines} sent={sent} b2b={b2b} combo={combo} "
                      f"| total sent {sent_total}, {pps:.2f} PPS, fixes {corrections}, "
                      f"pre-thought {reused}      ", end="\r")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            ctx.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "play":
        play(use_hold="nohold" not in sys.argv[2:])
    elif cmd == "image" and len(sys.argv) > 2:
        debug_image(sys.argv[2])
    else:
        {"calibrate": calibrate, "debug": debug, "keytest": keytest}.get(cmd, lambda: print(__doc__))()
