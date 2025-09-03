import math
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.token import TokenState

COLOR_MAP = {
    Colors.RED: (220, 50, 50),
    Colors.GREEN: (60, 160, 60),
    Colors.YELLOW: (235, 200, 40),
    Colors.BLUE: (50, 90, 200),
}

SAFE_COLOR = (240, 240, 240)
STAR_COLOR = (255, 255, 255)
BG_COLOR = (245, 245, 245)
PATH_COLOR = (210, 210, 210)
HOME_COLUMN_COLOR = (255, 255, 255)

FONT = None
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception:
    pass

CELL = 40
PADDING = 20
RADIUS = 12

# Precompute ring coordinates for 52 positions in a loop
# We'll layout approximate circular path


def _polar(center: Tuple[int, int], r: float, theta: float) -> Tuple[int, int]:
    return int(center[0] + r * math.cos(theta)), int(center[1] + r * math.sin(theta))


def compute_main_path() -> List[Tuple[int, int]]:
    center = (PADDING + 13 * CELL // 2, PADDING + 13 * CELL // 2)
    radius = 5 * CELL
    coords = []
    for i in range(52):
        theta = 2 * math.pi * (i / 52.0)
        coords.append(_polar(center, radius, theta))
    return coords


MAIN_PATH_COORDS = compute_main_path()


def draw_board(tokens: Dict[str, List[Dict]]) -> Image.Image:
    size = (PADDING * 2 + 13 * CELL, PADDING * 2 + 13 * CELL)
    img = Image.new("RGB", size, BG_COLOR)
    d = ImageDraw.Draw(img)

    # Draw main path nodes
    for idx, (x, y) in enumerate(MAIN_PATH_COORDS):
        r = 16
        pos_color = PATH_COLOR
        if idx in BoardConstants.STAR_SQUARES:
            pos_color = STAR_COLOR
        d.ellipse((x - r, y - r, x + r, y + r), fill=pos_color, outline=(180, 180, 180))

    # Draw home column slots (simple lines inward for each color)
    for color, entry in BoardConstants.HOME_COLUMN_ENTRIES.items():
        entry_xy = MAIN_PATH_COORDS[entry]
        cx, cy = entry_xy
        for step in range(1, GameConstants.HOME_COLUMN_SIZE + 1):
            t = step / (GameConstants.HOME_COLUMN_SIZE + 1)
            inner = (int(cx + (size[0] / 2 - cx) * t), int(cy + (size[1] / 2 - cy) * t))
            d.line([(cx, cy), inner], fill=COLOR_MAP[color], width=2)

    # Draw tokens
    for color, token_list in tokens.items():
        for tk in token_list:
            state = tk["state"]
            pos = tk["position"]
            if state == TokenState.HOME.value:
                # stack home tokens around outer corner of color start
                base = MAIN_PATH_COORDS[BoardConstants.START_POSITIONS[color]]
                # offset ring
                offsets = [(0, 0), (18, 0), (0, 18), (18, 18)]
                home_idx = tk["token_id"] % 4
                x = base[0] + offsets[home_idx][0] - 10
                y = base[1] + offsets[home_idx][1] - 10
            elif state == TokenState.FINISHED.value:
                x = size[0] // 2 + (tk["token_id"] % 2) * 22 - 20
                y = size[1] // 2 + (tk["token_id"] // 2) * 22 - 20
            elif state == TokenState.HOME_COLUMN.value:
                # interpolate from entry to center
                entry = BoardConstants.HOME_COLUMN_ENTRIES[color]
                cx, cy = MAIN_PATH_COORDS[entry]
                center = (size[0] // 2, size[1] // 2)
                depth = pos - BoardConstants.HOME_COLUMN_START + 1
                t = depth / (GameConstants.HOME_COLUMN_SIZE + 1)
                x = int(cx + (center[0] - cx) * t) - 10
                y = int(cy + (center[1] - cy) * t) - 10
            else:  # active
                if 0 <= pos < len(MAIN_PATH_COORDS):
                    x, y = MAIN_PATH_COORDS[pos]
                    x -= 10
                    y -= 10
                else:
                    continue
            d.ellipse((x, y, x + 20, y + 20), fill=COLOR_MAP[color], outline=(0, 0, 0))
            if FONT:
                d.text((x + 6, y + 4), str(tk["token_id"]), fill=(0, 0, 0), font=FONT)
    # Center finish circle
    d.ellipse(
        (size[0] // 2 - 30, size[1] // 2 - 30, size[0] // 2 + 30, size[1] // 2 + 30),
        outline=(80, 80, 80),
        width=3,
    )
    return img
