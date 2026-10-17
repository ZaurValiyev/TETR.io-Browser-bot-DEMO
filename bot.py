"""Educational TETR.IO bot: Playwright (input) + OpenCV (vision) + heuristic AI.

Usage:
    python bot.py calibrate   # open browser, drag a box around the 10x20 matrix
    python bot.py debug       # print what the bot "sees" and save debug.png
    python bot.py play        # play (Ctrl+C in this terminal to stop)

Practice use only (Zen / solo / private custom games). Do not use in ranked,
Quick Play or public lobbies; TETR.IO's rules forbid bots there.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from tetris_ai import COLS, ROWS, best_placement, identify_piece, key_plan

CONFIG = Path("config.json")
PROFILE_DIR = Path("browser_profile")  # keeps your login between runs
VIEWPORT = {"width": 1280, "height": 720}

# ---- Tunables -------------------------------------------------------------
BUFFER_ROWS = 2          # extra rows read ABOVE the matrix (pieces spawn there)
TOP_MASK_ROWS = 3        # matrix rows at the top treated as "falling piece" area
SAT_MIN, VAL_MIN = 70, 90  # HSV thresholds for "this cell is filled"
KEY_DELAY = 0.012        # seconds between key taps
POST_DROP_WAIT = 0.20    # let the piece lock and the next one spawn


# ---- Vision ---------------------------------------------------------------
def grab(page) -> np.ndarray:
    png = page.screenshot()
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def read_grid(img, roi):
    """Read the board as a grid of 0/1 with BUFFER_ROWS extra rows on top.

    Cells are square: size = box width / 10. Rows are counted UP from the box's
    bottom edge, so it doesn't matter if the box extends above the matrix.
    Returns BUFFER_ROWS + ROWS rows; index BUFFER_ROWS is the top matrix row.
    """
    x, y, w, h = roi
    cell = w / COLS
    bottom = y + h
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = BUFFER_ROWS + ROWS
    grid = [[0] * COLS for _ in range(total)]
    for i in range(total):
        cy = int(bottom - (total - i - 0.5) * cell)
        if cy < 3 or cy > img.shape[0] - 4:
            continue
        for c in range(COLS):
            cx = int(x + (c + 0.5) * cell)
            hp = hsv[cy - 2 : cy + 3, cx - 2 : cx + 3].reshape(-1, 3)
            _, s, v = np.median(hp, axis=0)
            if s > SAT_MIN and v > VAL_MIN:
                grid[i][c] = 1
    return grid


def spawn_cells(grid):
    """Filled cells in the spawn area: buffer rows plus the top matrix rows."""
    last = BUFFER_ROWS + TOP_MASK_ROWS
    return [(r, c) for r in range(last) for c in range(COLS) if grid[r][c]]


def detect_piece(grid):
    """Identify the falling piece by SHAPE. Returns (piece, box_x) or None."""
    return identify_piece(spawn_cells(grid))


def matrix_for_ai(grid):
    """The 20 matrix rows, with the falling-piece area cleared."""
    board = [row[:] for row in grid[BUFFER_ROWS:]]
    for r in range(TOP_MASK_ROWS):
        board[r] = [0] * COLS
    return board


def ascii_board(grid, piece):
    lines = [f"piece: {piece}   (^ = spawn area, ~ = above matrix)"]
    for i, row in enumerate(grid):
        tag = "~" if i < BUFFER_ROWS else ("^" if i < BUFFER_ROWS + TOP_MASK_ROWS else " ")
        lines.append(tag + "".join("#" if v else "." for v in row))
    return "\n".join(lines)


# ---- Browser --------------------------------------------------------------
def open_game(p):
    ctx = p.chromium.launch_persistent_context(
        str(PROFILE_DIR), headless=False, viewport=VIEWPORT, device_scale_factor=1
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://tetr.io/")
    return ctx, page


def load_roi():
    if not CONFIG.exists():
        sys.exit("No config.json yet: run `python bot.py calibrate` first.")
    return tuple(json.loads(CONFIG.read_text())["roi"])


def calibrate():
    with sync_playwright() as p:
        ctx, page = open_game(p)
        input("Log in / start a Zen or solo game so the matrix is visible, then press Enter here...")
        img = grab(page)
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
        input("Start a game, then press Enter to capture one frame...")
        img = grab(page)
        grid = read_grid(img, roi)
        print(ascii_board(grid, detect_piece(grid)))
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cell = w / COLS  # draw the cell centres that are actually sampled
        total = BUFFER_ROWS + ROWS
        for i in range(total):
            for c in range(COLS):
                px = int(x + (c + 0.5) * cell)
                py = int(y + h - (total - i - 0.5) * cell)
                col = (0, 0, 255) if grid[i][c] else (255, 255, 0)
                cv2.circle(img, (px, py), 2, col, -1)
        cv2.imwrite("debug.png", img)
        print("Saved debug.png (green box = region being read)")
        ctx.close()


def play():
    roi = load_roi()
    with sync_playwright() as p:
        ctx, page = open_game(p)
        input("Start a Zen/solo game and wait for it to begin, then press Enter to start the bot...")
        placed = 0
        try:
            while True:
                grid = read_grid(grab(page), roi)
                found = detect_piece(grid)
                if found is None:
                    print(f"no piece (spawn-area cells: {len(spawn_cells(grid))})    ", end="\r")
                    time.sleep(0.03)
                    continue
                piece, cur_x = found
                move = best_placement(matrix_for_ai(grid), piece)
                if move is None:
                    time.sleep(0.2)
                    continue
                rot, bx, _ = move
                for key in key_plan(rot, bx, cur_x):
                    page.keyboard.press(key)
                    time.sleep(KEY_DELAY)
                placed += 1
                print(f"#{placed}: {piece} rot={rot} x={bx}")
                time.sleep(POST_DROP_WAIT)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            ctx.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"calibrate": calibrate, "debug": debug, "play": play}.get(
        cmd, lambda: print(__doc__)
    )()
