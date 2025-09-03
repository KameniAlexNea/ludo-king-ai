import math
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.token import TokenState

# Styling
COLOR_MAP = {
    Colors.RED: (230, 60, 60),
    Colors.GREEN: (60, 170, 90),
    Colors.YELLOW: (245, 205, 55),
    Colors.BLUE: (65, 100, 210),
}
BG_COLOR = (245, 245, 245)
GRID_LINE = (210, 210, 210)
PATH_COLOR = (255, 255, 255)
STAR_COLOR = (255, 255, 200)
HOME_SHADE = (235, 235, 235)
CENTER_COLOR = (255, 255, 255)

FONT = None
try:  # optional font
    FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception:
    pass

# Basic geometric layout (15x15 grid for classic style)
CELL = 32
GRID = 15
BOARD_SIZE = GRID * CELL

# Derived constants
MAIN_SIZE = BoardConstants.STAR_SQUARES.__class__  # just to silence linter if unused
HOME_COLUMN_START = GameConstants.HOME_COLUMN_START
HOME_COLUMN_END = GameConstants.FINISH_POSITION
HOME_COLUMN_SIZE = GameConstants.HOME_COLUMN_SIZE

# We derive path coordinates procedurally using a canonical 52-step outer path.
# Layout: Imagine a cross with a 3-wide corridor. We'll build a ring path list of (col,row).

def _build_path_grid() -> List[Tuple[int, int]]:
    # Manual procedural trace of standard 52 cells referencing a 15x15 layout.
    # Start from (6,0) and move clockwise replicating earlier static mapping but generated.
    seq = []
    # Up column from (6,0)->(6,5)
    for r in range(0, 6):
        seq.append((6, r))
    # Left row (5,6)->(0,6)
    for c in range(5, -1, -1):
        seq.append((c, 6))
    # Down column (0,7)->(0,8)
    for r in range(7, 9):
        seq.append((0, r))
    # Right row (1,8)->(5,8)
    for c in range(1, 6):
        seq.append((c, 8))
    # Down column (6,9)->(6,14)
    for r in range(9, 15):
        seq.append((6, r))
    # Right row (7,14)->(8,14)
    for c in range(7, 9):
        seq.append((c, 14))
    # Up column (8,13)->(8,9)
    for r in range(13, 8, -1):
        seq.append((8, r))
    # Right row (9,8)->(14,8)
    for c in range(9, 15):
        seq.append((c, 8))
    # Up column (14,7)->(14,6)
    for r in range(7, 5, -1):
        seq.append((14, r))
    # Left row (13,6)->(9,6)
    for c in range(13, 8, -1):
        seq.append((c, 6))
    # Up column (8,5)->(8,0)
    for r in range(5, -1, -1):
        seq.append((8, r))
    # Left row (7,0)
    seq.append((7, 0))
    # Ensure length 52
    return seq

PATH_LIST = _build_path_grid()
PATH_INDEX_TO_COORD = {i: coord for i, coord in enumerate(PATH_LIST)}

# Home quadrants bounding boxes (col range inclusive)
HOME_QUADRANTS = {
    Colors.RED: ((0, 5), (0, 5)),
    Colors.GREEN: ((9, 14), (0, 5)),
    Colors.YELLOW: ((0, 5), (9, 14)),
    Colors.BLUE: ((9, 14), (9, 14)),
}

def _cell_bbox(col: int, row: int):
    x0 = col * CELL
    y0 = row * CELL
    return (x0, y0, x0 + CELL, y0 + CELL)

def _draw_home_quadrants(d: ImageDraw.ImageDraw):
    for color, ((c0, c1), (r0, r1)) in HOME_QUADRANTS.items():
        box = (c0 * CELL, r0 * CELL, (c1 + 1) * CELL, (r1 + 1) * CELL)
        d.rectangle(box, fill=tuple(int(c * 0.9) for c in COLOR_MAP[color]))
        inner = ((c0 + 1) * CELL, (r0 + 1) * CELL, c1 * CELL, r1 * CELL)
        d.rectangle(inner, fill=HOME_SHADE)

def _token_home_grid_position(color: str, token_id: int) -> Tuple[int, int]:
    (c0, c1), (r0, r1) = HOME_QUADRANTS[color]
    cols = [c0 + 1, c0 + 3]
    rows = [r0 + 1, r0 + 3]
    col = cols[token_id % 2]
    row = rows[token_id // 2]
    return col, row

def _home_column_positions_for_color(color: str) -> Dict[int, Tuple[int, int]]:
    # Map logical home column indices (100..105) to coordinates along middle lane toward center (7,7)
    mapping: Dict[int, Tuple[int, int]] = {}
    center = (7, 7)
    entry_index = BoardConstants.HOME_COLUMN_ENTRIES[color]
    entry_coord = PATH_INDEX_TO_COORD[entry_index]
    ex, ey = entry_coord
    # Determine axis of approach: same col or same row progression towards center
    dx = 0 if ex == center[0] else (1 if center[0] > ex else -1)
    dy = 0 if ey == center[1] else (1 if center[1] > ey else -1)
    cx, cy = ex + dx, ey + dy
    for offset in range(GameConstants.HOME_COLUMN_SIZE):
        mapping[HOME_COLUMN_START + offset] = (cx, cy)
        cx += dx
        cy += dy
    return mapping

HOME_COLUMN_COORDS = {color: _home_column_positions_for_color(color) for color in Colors.ALL_COLORS}

def draw_board(tokens: Dict[str, List[Dict]]) -> Image.Image:
    img = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), BG_COLOR)
    d = ImageDraw.Draw(img)

    # Quadrants
    _draw_home_quadrants(d)

    # Main path cells
    for idx, (c, r) in PATH_INDEX_TO_COORD.items():
        bbox = _cell_bbox(c, r)
        fill = STAR_COLOR if idx in BoardConstants.STAR_SQUARES else PATH_COLOR
        d.rectangle(bbox, fill=fill, outline=GRID_LINE)

    # Home columns
    for color, pos_map in HOME_COLUMN_COORDS.items():
        col_rgb = COLOR_MAP[color]
        for pos, (c, r) in pos_map.items():
            bbox = _cell_bbox(c, r)
            d.rectangle(bbox, fill=PATH_COLOR, outline=col_rgb)

    # Center finish region
    center_bbox = _cell_bbox(7, 7)
    d.rectangle(center_bbox, fill=CENTER_COLOR, outline=(80, 80, 80), width=3)

    # Grid overlay (subtle)
    for i in range(GRID + 1):
        d.line((0, i * CELL, BOARD_SIZE, i * CELL), fill=(230, 230, 230))
        d.line((i * CELL, 0, i * CELL, BOARD_SIZE), fill=(230, 230, 230))

    # Tokens
    for color, tlist in tokens.items():
        base_color = COLOR_MAP[color]
        for tk in tlist:
            state = tk['state']
            pos = tk['position']
            tid = tk['token_id']
            if state == TokenState.HOME.value:
                c, r = _token_home_grid_position(color, tid)
            elif state == TokenState.HOME_COLUMN.value and HOME_COLUMN_START <= pos <= HOME_COLUMN_END:
                coord_map = HOME_COLUMN_COORDS[color]
                if pos not in coord_map:
                    continue
                c, r = coord_map[pos]
            elif state == TokenState.FINISHED.value:
                # finished tokens stack inside center (2x2)
                offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
                oc, orow = offsets[tid % 4]
                c = 7 + oc - 1
                r = 7 + orow - 1
            else:  # active on main path
                if 0 <= pos < len(PATH_INDEX_TO_COORD):
                    c, r = PATH_INDEX_TO_COORD[pos]
                else:
                    continue
            bbox = _cell_bbox(c, r)
            x0, y0, x1, y1 = bbox
            inset = 4
            token_box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
            d.ellipse(token_box, fill=base_color, outline=(0, 0, 0))
            if FONT:
                d.text((x0 + CELL // 2 - 5, y0 + CELL // 2 - 8), str(tid), fill=(0, 0, 0), font=FONT)

    return img
