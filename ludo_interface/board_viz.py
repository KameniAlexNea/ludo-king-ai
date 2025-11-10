from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.piece import Piece as Token

from .models import ALL_COLORS, HOME_ENTRY, PLAYER_START_SQUARES, PlayerColor

# Enhanced Styling with gradients and better colors
COLOR_MAP = {
    PlayerColor.RED: (220, 53, 69),
    PlayerColor.GREEN: (40, 167, 69),
    PlayerColor.YELLOW: (255, 193, 7),
    PlayerColor.BLUE: (13, 110, 253),
}

# Additional color variations for better visuals
COLOR_LIGHT = {
    PlayerColor.RED: (248, 215, 218),
    PlayerColor.GREEN: (209, 231, 221),
    PlayerColor.YELLOW: (255, 243, 205),
    PlayerColor.BLUE: (204, 229, 255),
}

COLOR_DARK = {
    PlayerColor.RED: (176, 42, 55),
    PlayerColor.GREEN: (32, 134, 55),
    PlayerColor.YELLOW: (204, 154, 6),
    PlayerColor.BLUE: (10, 88, 202),
}

BG_COLOR = (248, 249, 250)
GRID_LINE = (173, 181, 189)
PATH_COLOR = (255, 255, 255)
STAR_COLOR = (255, 235, 59)  # More attractive star color
HOME_SHADE = (233, 236, 239)
CENTER_COLOR = (248, 249, 250)
SHADOW_COLOR = (0, 0, 0, 50)  # Semi-transparent black for shadows

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
HOME_COLUMN_START = config.HOME_COLUMN_ENTRIES
HOME_COLUMN_END = config.HOME_COLUMN_ENTRIES + config.HOME_COLUMN_SIZE - 1
HOME_COLUMN_SIZE = config.HOME_COLUMN_SIZE

# Global board template cache
_BOARD_TEMPLATE = None

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
    PlayerColor.RED: ((0, 5), (0, 5)),  # top-left
    PlayerColor.GREEN: ((0, 5), (9, 14)),  # bottom-left
    PlayerColor.YELLOW: ((9, 14), (9, 14)),  # bottom-right
    PlayerColor.BLUE: ((9, 14), (0, 5)),  # top-right
}


def _cell_bbox(col: int, row: int):
    x0 = col * CELL
    y0 = row * CELL
    return (x0, y0, x0 + CELL, y0 + CELL)


def _draw_gradient_rect(d: ImageDraw.ImageDraw, bbox, color_start, color_end):
    """Draw a rectangle with vertical gradient."""
    x0, y0, x1, y1 = bbox
    height = y1 - y0

    if height <= 0:
        return

    for i in range(height):
        ratio = i / height
        r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
        g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
        b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
        color = (r, g, b)
        d.line([(x0, y0 + i), (x1 - 1, y0 + i)], fill=color)


def _draw_shadow(d: ImageDraw.ImageDraw, bbox, offset=2):
    """Draw a subtle shadow for the given bounding box."""
    x0, y0, x1, y1 = bbox
    shadow_bbox = (x0 + offset, y0 + offset, x1 + offset, y1 + offset)
    d.rectangle(shadow_bbox, fill=(200, 200, 200, 100), outline=None)


def _draw_home_quadrants(d: ImageDraw.ImageDraw):
    """Draw enhanced home quadrants with gradients and better styling."""
    border_width = 8
    for color, ((c0, c1), (r0, r1)) in HOME_QUADRANTS.items():
        box = (c0 * CELL, r0 * CELL, (c1 + 1) * CELL, (r1 + 1) * CELL)

        # Draw gradient background
        light_color = COLOR_LIGHT[color]
        main_color = COLOR_MAP[color]
        _draw_gradient_rect(d, box, light_color, main_color + (50,))

        # Draw enhanced border with inner shadow effect
        for w in range(border_width):
            alpha = int(255 * (1 - w / border_width) * 0.8)
            border_color = COLOR_DARK[color] + (alpha,)
            inset_box = (
                box[0] + w,
                box[1] + w,
                box[2] - w,
                box[3] - w,
            )
            d.rectangle(inset_box, outline=border_color, width=2)


def _draw_stacked_tokens(
    d: ImageDraw.ImageDraw,
    tokens_at_pos: List[Tuple[str, Token]],
    center_x: int,
    center_y: int,
    base_radius: int,
    show_ids: bool = True,
):
    """Draw multiple tokens stacked at the same position with offset and shadow effects."""
    if not tokens_at_pos:
        return

    # Calculate stacking offsets
    num_tokens = len(tokens_at_pos)
    if num_tokens == 1:
        offsets = [(0, 0)]
    elif num_tokens == 2:
        offsets = [(-6, -6), (6, 6)]
    elif num_tokens == 3:
        offsets = [(-8, -8), (0, 0), (8, 8)]
    else:  # 4 or more tokens
        offsets = [(-8, -8), (8, -8), (-8, 8), (8, 8)]
        # If more than 4, stack them on top
        for i in range(4, num_tokens):
            offsets.append((0, 0))

    # Draw tokens with shadows and stacking
    for i, ((color, token), offset) in enumerate(zip(tokens_at_pos, offsets)):
        offset_x, offset_y = offset
        token_x = center_x + offset_x
        token_y = center_y + offset_y

        # Draw shadow first
        shadow_radius = base_radius + 2
        shadow_x = token_x + 2
        shadow_y = token_y + 2
        d.ellipse(
            (
                shadow_x - shadow_radius,
                shadow_y - shadow_radius,
                shadow_x + shadow_radius,
                shadow_y + shadow_radius,
            ),
            fill=(0, 0, 0, 80),
        )

        # Draw token with gradient effect
        token_color = COLOR_MAP[color]
        light_color = COLOR_LIGHT[color]

        # Outer ring (darker)
        d.ellipse(
            (
                token_x - base_radius,
                token_y - base_radius,
                token_x + base_radius,
                token_y + base_radius,
            ),
            fill=COLOR_DARK[color],
            outline=(0, 0, 0, 180),
            width=2,
        )

        # Inner circle (lighter)
        inner_radius = base_radius - 3
        d.ellipse(
            (
                token_x - inner_radius,
                token_y - inner_radius,
                token_x + inner_radius,
                token_y + inner_radius,
            ),
            fill=token_color,
        )

        # Highlight (top-left)
        highlight_radius = base_radius - 6
        highlight_x = token_x - 2
        highlight_y = token_y - 2
        d.ellipse(
            (
                highlight_x - highlight_radius,
                highlight_y - highlight_radius,
                highlight_x + highlight_radius,
                highlight_y + highlight_radius,
            ),
            fill=light_color,
        )

        # Token ID
        if show_ids and FONT:
            text_color = (255, 255, 255) if sum(token_color) < 400 else (0, 0, 0)
            d.text(
                (token_x - 5, token_y - 8),
                str(token.piece_id),
                fill=text_color,
                font=FONT,
            )

        # Stack indicator for multiple tokens
        if num_tokens > 1:
            stack_indicator_x = token_x + base_radius - 6
            stack_indicator_y = token_y - base_radius + 2
            d.ellipse(
                (
                    stack_indicator_x - 4,
                    stack_indicator_y - 4,
                    stack_indicator_x + 4,
                    stack_indicator_y + 4,
                ),
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=1,
            )
            if FONT:
                d.text(
                    (stack_indicator_x - 3, stack_indicator_y - 6),
                    str(num_tokens),
                    fill=(0, 0, 0),
                    font=FONT,
                )


def _token_home_grid_position(color: PlayerColor, token_id: int) -> Tuple[int, int]:
    (c0, c1), (r0, r1) = HOME_QUADRANTS[color]
    cols = [c0 + 1, c0 + 3]
    rows = [r0 + 1, r0 + 3]
    col = cols[token_id % 2]
    row = rows[token_id // 2]
    return col, row


def _home_column_positions_for_color(color: PlayerColor) -> Dict[int, Tuple[int, int]]:
    """
    Map home column indices (52..56) to board coordinates; 57 is final finish.

    GameConstants.HOME_COLUMN_SIZE = 6 covers 52..57 inclusive, but per spec 57 is
    not a drawable lane square—tokens reaching 57 are considered finished and moved
    to the center aggregation. We therefore only allocate 5 visual squares (52-56).
    """
    mapping: Dict[int, Tuple[int, int]] = {}
    center = (7, 7)
    entry_index = HOME_ENTRY[color]
    entry_coord = PATH_INDEX_TO_COORD[entry_index]
    ex, ey = entry_coord
    dx = 0 if ex == center[0] else (1 if center[0] > ex else -1)
    dy = 0 if ey == center[1] else (1 if center[1] > ey else -1)
    cx, cy = ex + dx, ey + dy
    # Only create squares for 100..104 (size - 1)
    for offset in range(config.HOME_COLUMN_SIZE - 1):  # exclude final 105
        mapping[HOME_COLUMN_START + offset] = (cx, cy)
        cx += dx
        cy += dy
    return mapping


HOME_COLUMN_COORDS = {
    color: _home_column_positions_for_color(color) for color in ALL_COLORS
}

START_POSITIONS = {color: PLAYER_START_SQUARES[color] for color in ALL_COLORS}
HOME_COLUMN_ENTRIES = {color: HOME_ENTRY[color] for color in ALL_COLORS}


def _generate_board_template() -> Image.Image:
    """
    Generate the static board template (without tokens) that can be reused.
    This is cached and only generated once for performance optimization.
    """
    img = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), BG_COLOR)
    d = ImageDraw.Draw(img)

    # Quadrants with enhanced styling
    _draw_home_quadrants(d)

    # Precompute special colored squares: start positions & home entry positions
    start_positions = START_POSITIONS  # color -> index
    home_entries = HOME_COLUMN_ENTRIES  # color -> index
    start_index_to_color = {idx: clr for clr, idx in start_positions.items()}
    entry_index_to_color = {idx: clr for clr, idx in home_entries.items()}

    # Main path cells with enhanced coloring and shadows
    for idx, (c, r) in PATH_INDEX_TO_COORD.items():
        bbox = _cell_bbox(c, r)
        outline = GRID_LINE

        if idx in start_index_to_color:  # starting squares (safe)
            color = start_index_to_color[idx]
            fill = COLOR_MAP[color]
            # Add subtle shadow for start positions
            _draw_shadow(d, bbox, offset=1)
            d.rectangle(bbox, fill=fill, outline=COLOR_DARK[color], width=2)
            # Add star symbol for start positions
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            star_size = 8
            # Simple star shape using lines
            d.line(
                [(cx, cy - star_size), (cx, cy + star_size)],
                fill=(255, 255, 255),
                width=2,
            )
            d.line(
                [(cx - star_size, cy), (cx + star_size, cy)],
                fill=(255, 255, 255),
                width=2,
            )
            d.line(
                [
                    (cx - star_size // 2, cy - star_size // 2),
                    (cx + star_size // 2, cy + star_size // 2),
                ],
                fill=(255, 255, 255),
                width=2,
            )
            d.line(
                [
                    (cx - star_size // 2, cy + star_size // 2),
                    (cx + star_size // 2, cy - star_size // 2),
                ],
                fill=(255, 255, 255),
                width=2,
            )
        elif idx in entry_index_to_color:  # home entry squares
            color = entry_index_to_color[idx]
            fill = PATH_COLOR
            outline = COLOR_MAP[color]
            d.rectangle(bbox, fill=fill, outline=outline, width=3)
        elif idx in config.STAR_SQUARES:  # global safe/star
            fill = STAR_COLOR
            d.rectangle(bbox, fill=fill, outline=outline, width=2)
            # Add star decoration
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            star_size = 6
            d.polygon(
                [
                    (cx, cy - star_size),
                    (cx + 2, cy - 2),
                    (cx + star_size, cy),
                    (cx + 2, cy + 2),
                    (cx, cy + star_size),
                    (cx - 2, cy + 2),
                    (cx - star_size, cy),
                    (cx - 2, cy - 2),
                ],
                fill=(255, 165, 0),
                outline=(0, 0, 0),
            )
        else:
            fill = PATH_COLOR
            d.rectangle(bbox, fill=fill, outline=outline)

    # Home columns with enhanced styling
    for color, pos_map in HOME_COLUMN_COORDS.items():
        base_color = COLOR_MAP[color]
        light_color = COLOR_LIGHT[color]
        for pos, (c, r) in pos_map.items():
            bbox = _cell_bbox(c, r)
            _draw_gradient_rect(d, bbox, light_color, base_color)
            d.rectangle(bbox, fill=None, outline=COLOR_DARK[color], width=2)

    # Enhanced center finish region
    cx0, cy0, cx1, cy1 = _cell_bbox(7, 7)
    midx = (cx0 + cx1) // 2
    midy = (cy0 + cy1) // 2

    # Shadow for center
    _draw_shadow(d, (cx0, cy0, cx1, cy1), offset=3)

    # Center background
    d.rectangle((cx0, cy0, cx1, cy1), fill=CENTER_COLOR, outline=(60, 60, 60), width=4)

    # Enhanced triangles with gradients
    colors_order = ALL_COLORS
    triangle_coords = [
        [(cx0, cy0), (cx1, cy0), (midx, midy)],  # top
        [(cx1, cy0), (cx1, cy1), (midx, midy)],  # right
        [(cx0, cy1), (cx1, cy1), (midx, midy)],  # bottom
        [(cx0, cy0), (cx0, cy1), (midx, midy)],  # left
    ]

    for i, (color, coords) in enumerate(zip(colors_order, triangle_coords)):
        d.polygon(coords, fill=COLOR_MAP[color], outline=COLOR_DARK[color])

    # Enhanced grid overlay
    for i in range(GRID + 1):
        alpha = 100 if i % 3 == 0 else 50  # Stronger lines every 3 cells
        grid_color = (200, 200, 200, alpha)
        d.line((0, i * CELL, BOARD_SIZE, i * CELL), fill=grid_color)
        d.line((i * CELL, 0, i * CELL, BOARD_SIZE), fill=grid_color)

    return img


def get_board_template() -> Image.Image:
    """
    Get the cached board template, generating it if necessary.
    Returns a copy to prevent modifications to the template.
    """
    global _BOARD_TEMPLATE
    if _BOARD_TEMPLATE is None:
        _BOARD_TEMPLATE = _generate_board_template()
    return _BOARD_TEMPLATE.copy()


def draw_board(
    tokens: Dict[PlayerColor, List[Token]], show_ids: bool = True
) -> Image.Image:
    """
    Optimized board drawing that uses a cached template and only draws tokens.
    This significantly improves performance by avoiding regenerating the board layout.
    """
    # Start with the cached board template
    img = get_board_template()
    d = ImageDraw.Draw(img)

    # Calculate finish anchors (same as in template generation)
    cx0, cy0, cx1, cy1 = _cell_bbox(7, 7)
    midx = (cx0 + cx1) // 2
    midy = (cy0 + cy1) // 2
    finish_anchor = {
        PlayerColor.RED: (midx, cy0 + (midy - cy0) // 2),
        PlayerColor.BLUE: (cx1 - (cx1 - midx) // 2, midy),
        PlayerColor.YELLOW: (midx, cy1 - (cy1 - midy) // 2),
        PlayerColor.GREEN: (cx0 + (midx - cx0) // 2, midy),
    }

    # Enhanced token rendering with proper stacking
    # First, collect all tokens by position and state for stacking
    position_groups: dict[str, list[Token]] = {}

    for color, tlist in tokens.items():
        for tk in tlist:
            rel = int(tk.position)

            if rel == 0:
                # Home tokens - render individually in their designated spots
                c, r = _token_home_grid_position(color, tk.piece_id)
                bbox = _cell_bbox(c, r)
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                _draw_stacked_tokens(d, [(color, tk)], cx, cy, CELL // 2 - 6, show_ids)

            elif HOME_COLUMN_START <= rel < HOME_COLUMN_END:
                # Home column tokens
                coord_map = HOME_COLUMN_COORDS[color]
                if rel in coord_map:
                    key = f"home_column_{color.name}_{rel}"
                    if key not in position_groups:
                        position_groups[key] = []
                    position_groups[key].append((color, tk))

            elif rel == config.HOME_FINISH:
                # Finished tokens - stack at finish anchor
                key = f"finished_{color.name}"
                if key not in position_groups:
                    position_groups[key] = []
                position_groups[key].append((color, tk))

            else:  # active on main path (relative ring 1..51)
                abs_idx = (START_POSITIONS[color] + rel - 1) % 52
                if 0 <= abs_idx < len(PATH_INDEX_TO_COORD):
                    key = f"main_path_{abs_idx}"
                    if key not in position_groups:
                        position_groups[key] = []
                    position_groups[key].append((color, tk))

    # Render grouped tokens with proper stacking
    for key, token_group in position_groups.items():
        if key.startswith("home_column_"):
            parts = key.split("_")
            color = PlayerColor[parts[2]]
            pos = int(parts[3])
            coord_map = HOME_COLUMN_COORDS[color]
            if pos in coord_map:
                c, r = coord_map[pos]
                bbox = _cell_bbox(c, r)
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                _draw_stacked_tokens(d, token_group, cx, cy, CELL // 2 - 4, show_ids)

        elif key.startswith("finished_"):
            color = PlayerColor[key.split("_")[1]]
            ax, ay = finish_anchor[color]
            _draw_stacked_tokens(d, token_group, ax, ay, CELL // 2 - 4, show_ids)

        elif key.startswith("main_path_"):
            pos = int(key.split("_")[2])
            c, r = PATH_INDEX_TO_COORD[pos]
            bbox = _cell_bbox(c, r)
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            _draw_stacked_tokens(d, token_group, cx, cy, CELL // 2 - 4, show_ids)

    return img


def clear_board_cache():
    """
    Clear the board template cache.
    Useful if you want to regenerate the template (e.g., after changing styling).
    """
    global _BOARD_TEMPLATE
    _BOARD_TEMPLATE = None


def preload_board_template():
    """
    Preload the board template to ensure the first draw is fast.
    Useful to call this during application startup.
    """
    get_board_template()
    print("🎯 Board template preloaded and cached for optimal performance!")
