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

from tetris_ai import COLS, ROWS, best_placement, key_plan

CONFIG = Path("config.json")
PROFILE_DIR = Path("browser_profile")  # keeps your login between runs
VIEWPORT = {"width": 1280, "height": 720}

# ---- Tunables -------------------------------------------------------------
TOP_MASK_ROWS = 3        # rows at the top used to read the *falling* piece
SAT_MIN, VAL_MIN = 70, 90  # HSV thresholds for "this cell is filled"
KEY_DELAY = 0.012        # seconds between key taps
POST_DROP_WAIT = 0.20    # let the piece lock and the next one spawn
MAX_COLOR_DIST = 90      # RGB distance for a piece colour match

# Approximate TETR.IO default-skin colours (RGB). Adjust after `debug`.
PIECE_RGB = {
    "Z": (239, 98, 77),
    "L": (239, 149, 53),
    "O": (247, 211, 8),
    "S": (102, 198, 92),
    "I": (66, 175, 225),
    "J": (25, 131, 188),
    "T": (185, 75, 198),
}


# ---- Vision ---------------------------------------------------------------
def grab(page) -> np.ndarray:
    png = page.screenshot()
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def read_grid(img, roi):
    """Return (grid, rgb) where grid[r][c] is 0/1 and rgb[r][c] the cell colour."""
    x, y, w, h = roi
    cw, ch = w / COLS, h / ROWS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grid = [[0] * COLS for _ in range(ROWS)]
    rgb = [[None] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            cx, cy = int(x + (c + 0.5) * cw), int(y + (r + 0.5) * ch)
            hp = hsv[cy - 2 : cy + 3, cx - 2 : cx + 3].reshape(-1, 3)
            _, s, v = np.median(hp, axis=0)
            if s > SAT_MIN and v > VAL_MIN:
                grid[r][c] = 1
                bp = img[cy - 2 : cy + 3, cx - 2 : cx + 3].reshape(-1, 3)
                b, g, rr = np.median(bp, axis=0)
                rgb[r][c] = (rr, g, b)
    return grid, rgb


def detect_piece(rgb):
    """Identify the falling piece from the filled cells in the top rows."""
    cells = [rgb[r][c] for r in range(TOP_MASK_ROWS) for c in range(COLS) if rgb[r][c]]
    if len(cells) < 3:
        return None
    mean = np.mean(cells, axis=0)
    name, dist = min(
        ((p, float(np.linalg.norm(mean - np.array(col)))) for p, col in PIECE_RGB.items()),
        key=lambda t: t[1],
    )
    return name if dist < MAX_COLOR_DIST else None

def describe_top(rgb):
    cells = [rgb[r][c] for r in range(TOP_MASK_ROWS) for c in range(COLS) if rgb[r][c]]
    if not cells:
        return "no filled cells in top rows"
    mean = np.mean(cells, axis=0)
    dists = sorted((float(np.linalg.norm(mean - np.array(col))), p) for p, col in PIECE_RGB.items())
    return f"{len(cells)} cells, mean RGB={tuple(int(v) for v in mean)}, nearest: {dists[0][1]} d={dists[0][0]:.0f}, then {dists[1][1]} d={dists[1][0]:.0f}"

def ascii_board(grid, piece):
    lines = [f"piece: {piece}"]
    for r, row in enumerate(grid):
        tag = "^" if r < TOP_MASK_ROWS else " "
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
        grid, rgb = read_grid(img, roi)
        print(ascii_board(grid, detect_piece(rgb)))
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
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
                grid, rgb = read_grid(grab(page), roi)
                piece = detect_piece(rgb)
                if piece is None:
                    filled = sum(map(sum, grid))
                    print(describe_top(rgb) + "    ", end="\r")
                    time.sleep(0.03)
                    continue
                board = [row[:] for row in grid]
                for r in range(TOP_MASK_ROWS):  # hide the falling piece from the AI
                    board[r] = [0] * COLS
                move = best_placement(board, piece)
                if move is None:
                    time.sleep(0.2)
                    continue
                rot, bx, _ = move
                for key in key_plan(rot, bx):
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
