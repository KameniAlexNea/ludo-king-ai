from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ludo_engine.constants import BoardConstants, Colors, GameConstants
from ludo_engine.token import TokenState

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
STAR_COLOR = (190, 190, 190)  # safe/star
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
    # Reordered to follow counter-clockwise Red -> Green -> Yellow -> Blue
    Colors.RED: ((0, 5), (0, 5)),  # top-left
    Colors.GREEN: ((0, 5), (9, 14)),  # bottom-left
    Colors.YELLOW: ((9, 14), (9, 14)),  # bottom-right
    Colors.BLUE: ((9, 14), (0, 5)),  # top-right
}


def _cell_bbox(col: int, row: int):
    x0 = col * CELL
    y0 = row * CELL
    return (x0, y0, x0 + CELL, y0 + CELL)


def _draw_home_quadrants(d: ImageDraw.ImageDraw):
    """Draw only a colored border for each player's home quadrant."""
    border_width = 6
    for color, ((c0, c1), (r0, r1)) in HOME_QUADRANTS.items():
        box = (c0 * CELL, r0 * CELL, (c1 + 1) * CELL, (r1 + 1) * CELL)
        # Draw base (background) to ensure any prior drawings are covered
        d.rectangle(box, fill=BG_COLOR)
        # Pillow's rectangle outline draws centered on the edge; for a thicker
        # appearance we can draw multiple inset rectangles.
        for w in range(border_width):
            inset_box = (
                box[0] + w,
                box[1] + w,
                box[2] - w,
                box[3] - w,
            )
            d.rectangle(inset_box, outline=COLOR_MAP[color])


def _token_home_grid_position(color: str, token_id: int) -> Tuple[int, int]:
    (c0, c1), (r0, r1) = HOME_QUADRANTS[color]
    cols = [c0 + 1, c0 + 3]
    rows = [r0 + 1, r0 + 3]
    col = cols[token_id % 2]
    row = rows[token_id // 2]
    return col, row


def _home_column_positions_for_color(color: str) -> Dict[int, Tuple[int, int]]:
    """
    Map home column indices (100..104) to board coordinates; 105 is final finish.

    GameConstants.HOME_COLUMN_SIZE = 6 covers 100..105 inclusive, but per spec 105 is
    not a drawable lane square—tokens reaching 105 are considered finished and moved
    to the center aggregation. We therefore only allocate 5 visual squares (100-104).
    """
    mapping: Dict[int, Tuple[int, int]] = {}
    center = (7, 7)
    entry_index = BoardConstants.HOME_COLUMN_ENTRIES[color]
    entry_coord = PATH_INDEX_TO_COORD[entry_index]
    ex, ey = entry_coord
    dx = 0 if ex == center[0] else (1 if center[0] > ex else -1)
    dy = 0 if ey == center[1] else (1 if center[1] > ey else -1)
    cx, cy = ex + dx, ey + dy
    # Only create squares for 100..104 (size - 1)
    for offset in range(GameConstants.HOME_COLUMN_SIZE - 1):  # exclude final 105
        mapping[HOME_COLUMN_START + offset] = (cx, cy)
        cx += dx
        cy += dy
    return mapping


HOME_COLUMN_COORDS = {
    color: _home_column_positions_for_color(color) for color in Colors.ALL_COLORS
}


def draw_board(tokens: Dict[str, List[Dict]], show_ids: bool = True) -> Image.Image:
    img = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), BG_COLOR)
    d = ImageDraw.Draw(img)

    # Quadrants
    _draw_home_quadrants(d)

    # Precompute special colored squares: start positions & home entry positions
    start_positions = BoardConstants.START_POSITIONS  # color -> index
    home_entries = BoardConstants.HOME_COLUMN_ENTRIES  # color -> index
    start_index_to_color = {idx: clr for clr, idx in start_positions.items()}
    entry_index_to_color = {idx: clr for clr, idx in home_entries.items()}

    # Main path cells with coloring rules
    for idx, (c, r) in PATH_INDEX_TO_COORD.items():
        bbox = _cell_bbox(c, r)
        outline = GRID_LINE
        if idx in start_index_to_color:  # starting squares (safe)
            fill = COLOR_MAP[start_index_to_color[idx]]
        elif (
            idx in entry_index_to_color
        ):  # home entry squares (NOT safe) keep path color, colored outline
            fill = PATH_COLOR
            outline = COLOR_MAP[entry_index_to_color[idx]]
        elif idx in BoardConstants.STAR_SQUARES:  # global safe/star
            fill = STAR_COLOR
        else:
            fill = PATH_COLOR
        d.rectangle(bbox, fill=fill, outline=outline)

    # Home columns (tinted player color)
    for color, pos_map in HOME_COLUMN_COORDS.items():
        col_rgb = COLOR_MAP[color]
        tint = tuple(min(255, int(v * 1.15)) for v in col_rgb)  # light tint
        for pos, (c, r) in pos_map.items():
            bbox = _cell_bbox(c, r)
            d.rectangle(bbox, fill=tint, outline=col_rgb)

    # Center finish region (position 105) draw four color triangles
    cx0, cy0, cx1, cy1 = _cell_bbox(7, 7)
    midx = (cx0 + cx1) // 2
    midy = (cy0 + cy1) // 2
    d.rectangle((cx0, cy0, cx1, cy1), fill=CENTER_COLOR, outline=(80, 80, 80), width=3)
    # Triangles: top(red), right(green), bottom(yellow), left(blue) typical clockwise
    d.polygon([(cx0, cy0), (cx1, cy0), (midx, midy)], fill=COLOR_MAP[Colors.RED])  # top
    d.polygon(
        [(cx1, cy0), (cx1, cy1), (midx, midy)], fill=COLOR_MAP[Colors.BLUE]
    )  # right (swapped: was GREEN)
    d.polygon(
        [(cx0, cy1), (cx1, cy1), (midx, midy)], fill=COLOR_MAP[Colors.YELLOW]
    )  # bottom
    d.polygon(
        [(cx0, cy0), (cx0, cy1), (midx, midy)], fill=COLOR_MAP[Colors.GREEN]
    )  # left (swapped: was BLUE)

    # Finish anchors per color inside their triangle (stack tokens exactly here)
    finish_anchor = {
        Colors.RED: (midx, cy0 + (midy - cy0) // 2),
        # Swapped BLUE and GREEN to align with triangle color swap
        Colors.BLUE: (cx1 - (cx1 - midx) // 2, midy),  # right side now BLUE
        Colors.YELLOW: (midx, cy1 - (cy1 - midy) // 2),
        Colors.GREEN: (cx0 + (midx - cx0) // 2, midy),  # left side now GREEN
    }

    # Grid overlay (subtle)
    for i in range(GRID + 1):
        d.line((0, i * CELL, BOARD_SIZE, i * CELL), fill=(230, 230, 230))
        d.line((i * CELL, 0, i * CELL, BOARD_SIZE), fill=(230, 230, 230))

    # Tokens
    for color, tlist in tokens.items():
        base_color = COLOR_MAP[color]
        for tk in tlist:
            state = tk["state"]
            pos = tk["position"]
            tid = tk["token_id"]
            if state == TokenState.HOME.value:
                c, r = _token_home_grid_position(color, tid)
            elif (
                state == TokenState.HOME_COLUMN.value
                and HOME_COLUMN_START <= pos <= HOME_COLUMN_END
            ):
                coord_map = HOME_COLUMN_COORDS[color]
                if pos not in coord_map:
                    continue
                c, r = coord_map[pos]
            elif state == TokenState.FINISHED.value:
                ax, ay = finish_anchor[color]
                # Draw stacked (superposed) circle; tokens overlap fully
                r_pix = CELL // 2 - 4
                x0 = ax - r_pix
                y0 = ay - r_pix
                x1 = ax + r_pix
                y1 = ay + r_pix
                d.ellipse((x0, y0, x1, y1), fill=base_color, outline=(0, 0, 0))
                if show_ids and FONT:
                    d.text((ax - 5, ay - 8), str(tid), fill=(0, 0, 0), font=FONT)
                continue
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
            if show_ids and FONT:
                d.text(
                    (x0 + CELL // 2 - 5, y0 + CELL // 2 - 8),
                    str(tid),
                    fill=(0, 0, 0),
                    font=FONT,
                )

    return img
