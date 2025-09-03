import math
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.token import TokenState

# Color styling
COLOR_MAP = {
    Colors.RED: (230, 60, 60),
    Colors.GREEN: (60, 170, 90),
    Colors.YELLOW: (245, 205, 55),
    Colors.BLUE: (65, 100, 210),
}
BG_COLOR = (245, 245, 245)
GRID_LINE = (200, 200, 200)
PATH_COLOR = (255, 255, 255)
STAR_COLOR = (255, 255, 200)
HOME_SHADE = (235, 235, 235)
CENTER_COLOR = (255, 255, 255)

FONT = None
try:  # Best-effort font
    FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception:
    pass

CELL = 32
GRID = 15  # 15x15 classic style grid
BOARD_SIZE = GRID * CELL

# Path mapping (index -> (col,row)) approximating classic Ludo layout
PATH_GRID = {
    0: (6, 0), 1: (6, 1), 2: (6, 2), 3: (6, 3), 4: (6, 4), 5: (6, 5),
    6: (5, 6), 7: (4, 6), 8: (3, 6), 9: (2, 6), 10: (1, 6), 11: (0, 6),
    12: (0, 7), 13: (0, 8), 14: (1, 8), 15: (2, 8), 16: (3, 8), 17: (4, 8), 18: (5, 8),
    19: (6, 9), 20: (6, 10), 21: (6, 11), 22: (6, 12), 23: (6, 13), 24: (6, 14),
    25: (7, 14), 26: (8, 14), 27: (8, 13), 28: (8, 12), 29: (8, 11), 30: (8, 10), 31: (8, 9),
    32: (9, 8), 33: (10, 8), 34: (11, 8), 35: (12, 8), 36: (13, 8), 37: (14, 8),
    38: (14, 7), 39: (14, 6), 40: (13, 6), 41: (12, 6), 42: (11, 6), 43: (10, 6), 44: (9, 6),
    45: (8, 5), 46: (8, 4), 47: (8, 3), 48: (8, 2), 49: (8, 1), 50: (8, 0), 51: (7, 0),
}

# Home column (100-105) mapping for each color moving toward center (7,7)
HOME_COLUMN_MAP = {
    'red':   {100: (7,1), 101: (7,2), 102: (7,3), 103: (7,4), 104: (7,5), 105: (7,6)},
    'green': {100: (1,7), 101: (2,7), 102: (3,7), 103: (4,7), 104: (5,7), 105: (6,7)},
    'yellow':{100: (7,13),101: (7,12),102: (7,11),103: (7,10),104: (7,9),105: (7,8)},
    'blue':  {100: (13,7),101: (12,7),102: (11,7),103: (10,7),104: (9,7),105: (8,7)},
}

STAR_INDICES = BoardConstants.STAR_SQUARES

# Home quadrants (col range, row range, color)
HOME_QUADRANTS = {
    'red':   ((0,5),(0,5)),
    'green': ((9,14),(0,5)),
    'yellow':((0,5),(9,14)),
    'blue':  ((9,14),(9,14)),
}

def _cell_bbox(col:int,row:int):
    x0 = col*CELL; y0 = row*CELL
    return (x0, y0, x0+CELL, y0+CELL)

def _draw_home_quadrants(d:ImageDraw.ImageDraw):
    for color,(cols,rows) in HOME_QUADRANTS.items():
        (c0,c1),(r0,r1) = cols, rows
        box = (c0*CELL, r0*CELL, (c1+1)*CELL, (r1+1)*CELL)
        d.rectangle(box, fill=tuple(int(c*0.9) for c in COLOR_MAP[getattr(Colors,color.upper())]))
        # inner white square for token staging
        inner = ( (c0+1)*CELL, (r0+1)*CELL, (c1)*CELL, (r1)*CELL )
        d.rectangle(inner, fill=HOME_SHADE)

def _token_home_position(color:str, token_id:int):
    # Place tokens in 2x2 grid inside quadrant inner area
    (cols,rows) = HOME_QUADRANTS[color]
    c0,c1 = cols; r0,r1 = rows
    # inner grid coords
    grid_cols = [c0+1, c0+3]
    grid_rows = [r0+1, r0+3]
    gc = grid_cols[token_id % 2]
    gr = grid_rows[token_id // 2]
    return gc, gr

def draw_board(tokens: Dict[str, List[Dict]]) -> Image.Image:
    img = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), BG_COLOR)
    d = ImageDraw.Draw(img)

    # Draw home quadrants
    _draw_home_quadrants(d)

    # Draw path squares
    for idx,(c,r) in PATH_GRID.items():
        bbox = _cell_bbox(c,r)
        fill = PATH_COLOR
        if idx in STAR_INDICES:
            fill = STAR_COLOR
        d.rectangle(bbox, fill=fill, outline=GRID_LINE)

    # Draw home columns
    for color,map_ in HOME_COLUMN_MAP.items():
        col_rgb = COLOR_MAP[getattr(Colors,color.upper())]
        for pos,(c,r) in map_.items():
            bbox = _cell_bbox(c,r)
            d.rectangle(bbox, fill=PATH_COLOR, outline=col_rgb)

    # Center finishing square
    center_bbox = _cell_bbox(7,7)
    d.rectangle(center_bbox, fill=CENTER_COLOR, outline=(80,80,80), width=3)

    # Draw grid lines (optional subtle)
    for i in range(GRID+1):
        d.line((0,i*CELL, BOARD_SIZE, i*CELL), fill=(230,230,230))
        d.line((i*CELL,0, i*CELL, BOARD_SIZE), fill=(230,230,230))

    # Draw tokens
    for color, tlist in tokens.items():
        base_color = COLOR_MAP[getattr(Colors,color.upper())]
        for tk in tlist:
            state = tk['state']; pos = tk['position']; tid = tk['token_id']
            if state == TokenState.HOME.value:
                c,r = _token_home_position(color, tid)
            elif state == TokenState.FINISHED.value:
                # place in center arranged 2x2
                offsets = [(0,0),(1,0),(0,1),(1,1)]
                offc,offr = offsets[tid]
                c = 7 + offc - 1
                r = 7 + offr - 1
            elif state == TokenState.HOME_COLUMN.value and pos in HOME_COLUMN_MAP[color]:
                c,r = HOME_COLUMN_MAP[color][pos]
            elif 0 <= pos < 52 and pos in PATH_GRID:
                c,r = PATH_GRID[pos]
            else:
                continue
            bbox = _cell_bbox(c,r)
            x0,y0,x1,y1 = bbox
            inset = 4
            token_box = (x0+inset,y0+inset,x1-inset,y1-inset)
            d.ellipse(token_box, fill=base_color, outline=(0,0,0))
            if FONT:
                d.text((x0+CELL//2-5,y0+CELL//2-8), str(tid), fill=(0,0,0), font=FONT)

    return img
