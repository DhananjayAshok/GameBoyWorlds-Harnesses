import re
from typing import Iterable

import numpy as np

TILE_SIZE = 8
GRID_COLS = 20
GRID_ROWS = 18

LIGHT_BACKGROUND = 200
DARK_INK = 128

BOTTOM_BOX_INNER_TOP = 104
BOTTOM_BOX_INNER_BOTTOM = 136
BOTTOM_BOX_INNER_LEFT = 16
BOTTOM_BOX_INNER_RIGHT = 144
BORDER_DARK_FRACTION = 0.75
MAX_BORDER_COLS = 12


def _grey_frame(frame: np.ndarray) -> np.ndarray:
    return frame[:, :, 0] if frame.ndim == 3 else frame


def is_cursor_tile(raw_tile: np.ndarray) -> bool:
    """Return True if an 8x8 tile shows a right-pointing triangle menu cursor."""
    if raw_tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
        return False
    tile_2d = _grey_frame(raw_tile)
    if (tile_2d >= LIGHT_BACKGROUND).mean() < 0.5:
        return False
    dark = (tile_2d < DARK_INK).astype(int)
    total = int(dark.sum())
    if not (12 <= total <= 28):
        return False
    row_sums = dark.sum(axis=1).tolist()
    col_sums = dark.sum(axis=0).tolist()
    nz_cols = [i for i, v in enumerate(col_sums) if v > 0]
    if not (3 <= len(nz_cols) <= 6) or nz_cols[-1] - nz_cols[0] + 1 != len(nz_cols):
        return False
    col_vals = [col_sums[i] for i in nz_cols]
    if max(col_vals[:2]) < 5 or col_vals[-1] > 2:
        return False
    peak_idx = 0 if col_vals[0] >= col_vals[1] else 1
    if any(col_vals[i] < col_vals[i + 1] for i in range(peak_idx, len(col_vals) - 1)):
        return False
    if col_vals[peak_idx] <= col_vals[-1] + 2:
        return False
    nz_rows = [i for i, v in enumerate(row_sums) if v > 0]
    if not (5 <= len(nz_rows) <= 7):
        return False
    rmax = max(row_sums)
    if rmax < 4 or (row_sums[3] != rmax and row_sums[4] != rmax):
        return False
    if row_sums[0] > 1 and row_sums[7] > 1:
        return False
    for k in range(len(nz_rows) // 2):
        if abs(row_sums[nz_rows[k]] - row_sums[nz_rows[-1 - k]]) > 1:
            return False
    return True


def has_menu_cursor(frame: np.ndarray) -> bool:
    """Return True if any 8x8 grid tile on screen contains a right-pointing menu cursor."""
    img = _grey_frame(frame)
    max_ty = min(GRID_ROWS, img.shape[0] // TILE_SIZE)
    max_tx = min(GRID_COLS, img.shape[1] // TILE_SIZE)
    for ty in range(max_ty):
        y0 = ty * TILE_SIZE
        for tx in range(max_tx):
            x0 = tx * TILE_SIZE
            if is_cursor_tile(img[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]):
                return True
    return False


def has_battle_menu_box(frame: np.ndarray) -> bool:
    """Return True if the bottom box has an interior vertical menu border (command or move menu)."""
    img = _grey_frame(frame)
    if img.shape[0] < BOTTOM_BOX_INNER_BOTTOM or img.shape[1] < BOTTOM_BOX_INNER_RIGHT:
        return False
    interior = img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM,
                   BOTTOM_BOX_INNER_LEFT:BOTTOM_BOX_INNER_RIGHT]
    if (interior >= LIGHT_BACKGROUND).mean() < 0.5:
        return False
    col_dark = (interior < DARK_INK).mean(axis=0)
    n_border_cols = int((col_dark >= BORDER_DARK_FRACTION).sum())
    if 1 <= n_border_cols <= MAX_BORDER_COLS:
        return True
    for tx in (4, 8):
        x0 = tx * TILE_SIZE
        if all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, x0:x0 + TILE_SIZE])
               for ty in range(13, 17)):
            gutter = img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM, TILE_SIZE:x0]
            if gutter.size > 0 and float((gutter >= DIALOGUE_BOX_WHITE).mean()) >= 0.85:
                return True
    return False


def is_battle_decision_screen(frame: np.ndarray) -> bool:
    """Return True when the battle screen displays a command menu, move list, or cursor choice box."""
    return has_battle_menu_box(frame) or has_menu_cursor(frame)


def is_same_battle_frame(earlier: np.ndarray, later: np.ndarray) -> bool:
    """Return True if two battle frames are identical outside the bottom-right scroll-arrow corner."""
    if earlier.shape != later.shape:
        return False
    if np.array_equal(earlier, later):
        return True
    a, b = _grey_frame(earlier), _grey_frame(later)
    differ = a != b
    if np.any(differ[:120, :]) or np.any(differ[120:, :140]):
        return False
    return True


DIALOGUE_BOX_TOP = 96
DIALOGUE_BOX_BOTTOM = 144
DIALOGUE_BOX_CHECK_LEFT = 8
DIALOGUE_BOX_CHECK_RIGHT = 104
DIALOGUE_BOX_WHITE = 240
DIALOGUE_BOX_MIN_WHITE_FRACTION = 0.70

CUTSCENE_DIFF_THRESHOLD = 40
CUTSCENE_DIFF_MIN_PIXELS = 80


def has_bottom_dialogue_box(frame: np.ndarray) -> bool:
    """Return True if the bottom of the screen shows a bordered dialogue box (even if blank)."""
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < DIALOGUE_BOX_CHECK_RIGHT:
        return False
    inner = img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM,
                DIALOGUE_BOX_CHECK_LEFT:DIALOGUE_BOX_CHECK_RIGHT]
    if float((inner >= DIALOGUE_BOX_WHITE).mean()) < DIALOGUE_BOX_MIN_WHITE_FRACTION:
        return False
    top_border = img[DIALOGUE_BOX_TOP:BOTTOM_BOX_INNER_TOP,
                     DIALOGUE_BOX_CHECK_LEFT:DIALOGUE_BOX_CHECK_RIGHT]
    bottom_border = img[BOTTOM_BOX_INNER_BOTTOM:DIALOGUE_BOX_BOTTOM,
                        DIALOGUE_BOX_CHECK_LEFT:DIALOGUE_BOX_CHECK_RIGHT]
    left_border = img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM, :DIALOGUE_BOX_CHECK_LEFT]
    n_top_tiles = (DIALOGUE_BOX_CHECK_RIGHT - DIALOGUE_BOX_CHECK_LEFT) // TILE_SIZE
    n_side_tiles = (BOTTOM_BOX_INNER_BOTTOM - BOTTOM_BOX_INNER_TOP) // TILE_SIZE
    for i in range(n_top_tiles):
        sl = slice(i * TILE_SIZE, (i + 1) * TILE_SIZE)
        top_tile = top_border[:, sl]
        bot_tile = bottom_border[:, sl]
        if not ((top_tile < DARK_INK).any() and (top_tile >= LIGHT_BACKGROUND).any()):
            return False
        if not ((bot_tile < DARK_INK).any() and (bot_tile >= LIGHT_BACKGROUND).any()):
            return False
    for i in range(n_side_tiles):
        side_tile = left_border[i * TILE_SIZE:(i + 1) * TILE_SIZE, :]
        if not ((side_tile < DARK_INK).any() and (side_tile >= LIGHT_BACKGROUND).any()):
            return False
    return True


def is_cutscene_frame_changing(earlier: np.ndarray, later: np.ndarray) -> bool:
    """Return True if sprite movement, emote animation, or box transitions changed between frames."""
    if earlier is None or later is None or earlier.shape != later.shape:
        return False
    if np.array_equal(earlier, later):
        return False
    a = _grey_frame(earlier).astype(int)
    b = _grey_frame(later).astype(int)
    diff = np.abs(a - b) > CUTSCENE_DIFF_THRESHOLD
    diff[88:, 104:] = False
    return int(diff.sum()) >= CUTSCENE_DIFF_MIN_PIXELS


BANNER_TOP = 112
BANNER_INNER_TOP = 120
BANNER_INNER_BOTTOM = 136
BANNER_BOTTOM = 144
BANNER_MIN_LIGHT_FRACTION = 0.65

MODAL_MIN_SPAN_TILES = 5
MODAL_MIN_LIGHT_FRACTION = 0.50
MODAL_MIN_MARGIN_LIGHT_FRACTION = 0.75


def is_horizontal_border_tile(raw_tile: np.ndarray) -> bool:
    """Return True if an 8x8 tile has horizontal UI frame border lines."""
    if raw_tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
        return False
    tile = _grey_frame(raw_tile)
    if not ((tile < DARK_INK).any() and (tile >= LIGHT_BACKGROUND).any()):
        return False
    row_dark = (tile < DARK_INK).mean(axis=1)
    row_light = (tile >= LIGHT_BACKGROUND).mean(axis=1)
    return bool((row_dark >= 0.5).any() and (row_light >= 0.5).any())


def is_vertical_border_tile(raw_tile: np.ndarray) -> bool:
    """Return True if an 8x8 tile has vertical UI frame border lines."""
    if raw_tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
        return False
    tile = _grey_frame(raw_tile)
    if not ((tile < DARK_INK).any() and (tile >= LIGHT_BACKGROUND).any()):
        return False
    col_dark = (tile < DARK_INK).mean(axis=0)
    col_light = (tile >= LIGHT_BACKGROUND).mean(axis=0)
    return bool((col_dark >= 0.5).any() and (col_light >= 0.5).any())


def has_bottom_location_banner(frame: np.ndarray) -> bool:
    """Return True if a bottom-anchored location/area banner overlay is drawn on the screen."""
    img = _grey_frame(frame)
    if img.shape[0] < BANNER_BOTTOM or img.shape[1] < DIALOGUE_BOX_CHECK_RIGHT:
        return False
    inner = img[BANNER_INNER_TOP:BANNER_INNER_BOTTOM,
                BOTTOM_BOX_INNER_LEFT:DIALOGUE_BOX_CHECK_RIGHT]
    if float((inner >= LIGHT_BACKGROUND).mean()) < BANNER_MIN_LIGHT_FRACTION:
        return False
    top_border = img[BANNER_TOP:BANNER_INNER_TOP,
                     DIALOGUE_BOX_CHECK_LEFT:DIALOGUE_BOX_CHECK_RIGHT]
    bottom_border = img[BANNER_INNER_BOTTOM:BANNER_BOTTOM,
                        DIALOGUE_BOX_CHECK_LEFT:DIALOGUE_BOX_CHECK_RIGHT]
    left_border = img[BANNER_INNER_TOP:BANNER_INNER_BOTTOM, :DIALOGUE_BOX_CHECK_LEFT]
    n_top_tiles = (DIALOGUE_BOX_CHECK_RIGHT - DIALOGUE_BOX_CHECK_LEFT) // TILE_SIZE
    n_side_tiles = (BANNER_INNER_BOTTOM - BANNER_INNER_TOP) // TILE_SIZE
    for i in range(n_top_tiles):
        sl = slice(i * TILE_SIZE, (i + 1) * TILE_SIZE)
        if not (is_horizontal_border_tile(top_border[:, sl])
                and is_horizontal_border_tile(bottom_border[:, sl])):
            return False
    for i in range(n_side_tiles):
        if not is_vertical_border_tile(left_border[i * TILE_SIZE:(i + 1) * TILE_SIZE, :]):
            return False
    return True


def find_modal_preview_boxes(frame: np.ndarray):
    """Return (top_ty, bot_ty, left_tx, right_tx) tile bounds for enclosed modal preview boxes."""
    img = _grey_frame(frame)
    max_ty = min(GRID_ROWS, img.shape[0] // TILE_SIZE)
    max_tx = min(GRID_COLS, img.shape[1] // TILE_SIZE)
    if max_ty < GRID_ROWS or max_tx < GRID_COLS:
        return []
    tiles = [[img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
              for tx in range(max_tx)] for ty in range(max_ty)]
    h_border = [[is_horizontal_border_tile(tiles[ty][tx]) for tx in range(max_tx)] for ty in range(max_ty)]
    v_border = [[is_vertical_border_tile(tiles[ty][tx]) for tx in range(max_tx)] for ty in range(max_ty)]
    boxes = []
    for top_ty in range(1, 11):
        for bot_ty in range(top_ty + MODAL_MIN_SPAN_TILES, 16):
            for left_tx in range(1, 12):
                for right_tx in range(left_tx + MODAL_MIN_SPAN_TILES, 19):
                    if not all(h_border[top_ty][tx] and h_border[bot_ty][tx]
                               for tx in range(left_tx + 1, right_tx)):
                        continue
                    if not all(v_border[ty][left_tx] for ty in range(top_ty + 1, bot_ty)):
                        continue
                    r_end = min(bot_ty, 12) if right_tx >= 13 else bot_ty
                    if not all(v_border[ty][right_tx] for ty in range(top_ty + 1, r_end)):
                        continue
                    y0, y1 = (top_ty + 1) * TILE_SIZE, bot_ty * TILE_SIZE
                    x0, x1 = (left_tx + 1) * TILE_SIZE, right_tx * TILE_SIZE
                    inner = img[y0:y1, x0:x1]
                    if float((inner >= LIGHT_BACKGROUND).mean()) < MODAL_MIN_LIGHT_FRACTION:
                        continue
                    margin_mask = np.zeros(inner.shape, dtype=bool)
                    margin_mask[:TILE_SIZE, :] = True
                    margin_mask[-TILE_SIZE:, :] = True
                    margin_mask[:, :TILE_SIZE] = True
                    margin_mask[:, -TILE_SIZE:] = True
                    if float((inner[margin_mask] >= LIGHT_BACKGROUND).mean()) < MODAL_MIN_MARGIN_LIGHT_FRACTION:
                        continue
                    boxes.append((top_ty, bot_ty, left_tx, right_tx))
    return boxes


def has_modal_preview_box(frame: np.ndarray) -> bool:
    """Return True if an enclosed modal preview window is open over the overworld."""
    return bool(find_modal_preview_boxes(frame))


TOP_BOX_TOP = 0
TOP_BOX_INNER_TOP = 8
TOP_BOX_INNER_BOTTOM = 24
TOP_BOX_BOTTOM = 32
TOP_BOX_INNER_LEFT = 8
TOP_BOX_INNER_RIGHT = 152

BATTLE_UPPER_BOTTOM = 96
BATTLE_TOP_STRIP_BOTTOM = 16
BATTLE_MID_X = 80
BATTLE_MIN_UPPER_WHITE_FRACTION = 0.75
BATTLE_MIN_HALF_WHITE_FRACTION = 0.65


def has_top_dialogue_box(frame: np.ndarray) -> bool:
    """Return True if a full-width 4-tile-tall bordered box is drawn at the top of the screen."""
    img = _grey_frame(frame)
    if img.shape[0] < TOP_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    inner = img[TOP_BOX_INNER_TOP:TOP_BOX_INNER_BOTTOM,
                TOP_BOX_INNER_LEFT:TOP_BOX_INNER_RIGHT]
    if float((inner >= DIALOGUE_BOX_WHITE).mean()) < DIALOGUE_BOX_MIN_WHITE_FRACTION:
        return False
    for tx in range(1, GRID_COLS - 1):
        sl = slice(tx * TILE_SIZE, (tx + 1) * TILE_SIZE)
        if not (is_horizontal_border_tile(img[TOP_BOX_TOP:TOP_BOX_INNER_TOP, sl])
                and is_horizontal_border_tile(img[TOP_BOX_INNER_BOTTOM:TOP_BOX_BOTTOM, sl])):
            return False
    for ty in (1, 2):
        rsl = slice(ty * TILE_SIZE, (ty + 1) * TILE_SIZE)
        if not (is_vertical_border_tile(img[rsl, :TOP_BOX_INNER_LEFT])
                and is_vertical_border_tile(img[rsl, TOP_BOX_INNER_RIGHT:GRID_COLS * TILE_SIZE])):
            return False
    return True


def is_battle_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows a battle backdrop above a bottom battle box."""
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if has_top_dialogue_box(img) or has_modal_preview_box(img):
        return False
    upper = img[:BATTLE_UPPER_BOTTOM, :]
    if float((upper >= DIALOGUE_BOX_WHITE).mean()) < BATTLE_MIN_UPPER_WHITE_FRACTION:
        return False
    if float((img[:BATTLE_TOP_STRIP_BOTTOM, :] >= DIALOGUE_BOX_WHITE).mean()) < BATTLE_MIN_UPPER_WHITE_FRACTION:
        return False
    if float((upper[:, :BATTLE_MID_X] >= DIALOGUE_BOX_WHITE).mean()) < BATTLE_MIN_HALF_WHITE_FRACTION:
        return False
    if float((upper[:, BATTLE_MID_X:] >= DIALOGUE_BOX_WHITE).mean()) < BATTLE_MIN_HALF_WHITE_FRACTION:
        return False
    return True


def is_move_selection_screen(frame: np.ndarray) -> bool:
    """Return True when the 4-move selection menu and TYPE/PP overlay box are open in battle."""
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 4 * TILE_SIZE:5 * TILE_SIZE])
               for ty in range(13, 17)):
        return False
    if float((img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM, TILE_SIZE:4 * TILE_SIZE]
              >= DIALOGUE_BOX_WHITE).mean()) < 0.90:
        return False
    if not any(is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 5 * TILE_SIZE:6 * TILE_SIZE])
               for ty in range(13, 17)):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, :TILE_SIZE])
               and is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 10 * TILE_SIZE:11 * TILE_SIZE])
               for ty in range(8, 11)):
        return False
    if float((img[8 * TILE_SIZE:11 * TILE_SIZE, TILE_SIZE:10 * TILE_SIZE] >= LIGHT_BACKGROUND).mean()) < 0.65:
        return False
    return True


def is_battle_menu_crop(crop: np.ndarray) -> bool:
    """Return True if a bottom-box crop shows a battle command menu or move selection menu rather than dialogue."""
    if crop is None or crop.size == 0:
        return False
    img = _grey_frame(crop)
    h, w = img.shape[:2]
    max_ty = h // TILE_SIZE
    max_tx = w // TILE_SIZE
    for ty in range(max_ty):
        y0 = ty * TILE_SIZE
        for tx in range(max_tx):
            x0 = tx * TILE_SIZE
            if is_cursor_tile(img[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]):
                return True
    if h < 32 or w < 64:
        return False
    y0 = (h - 32) // 2 if h > 32 else 0
    inner = img[y0:y0 + 32, :]
    for x in range(16, w - 24, 2):
        strip = inner[:, x:x + TILE_SIZE]
        if all(is_vertical_border_tile(strip[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, :]) for ty in range(4)):
            left_pad = inner[:, max(0, x - 16):x]
            if left_pad.size > 0 and float((left_pad >= DIALOGUE_BOX_WHITE).mean()) >= 0.85:
                return True
    return False


def is_empty_battle_bag_menu(frame: np.ndarray) -> bool:
    """Return True when the in-battle item sub-menu is open with only CANCEL at row 1 and blank rows below."""
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if not is_cursor_tile(img[4 * TILE_SIZE:5 * TILE_SIZE, 5 * TILE_SIZE:6 * TILE_SIZE]):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 4 * TILE_SIZE:5 * TILE_SIZE])
               for ty in range(3, 11)):
        return False
    if not all(is_horizontal_border_tile(img[2 * TILE_SIZE:3 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
               for tx in range(5, 19)):
        return False
    if not bool((img[4 * TILE_SIZE:5 * TILE_SIZE, 6 * TILE_SIZE:12 * TILE_SIZE] < DARK_INK).any()):
        return False
    below = img[5 * TILE_SIZE:11 * TILE_SIZE, 5 * TILE_SIZE:18 * TILE_SIZE]
    return int((below < DARK_INK).sum()) == 0 and float((below >= DIALOGUE_BOX_WHITE).mean()) >= 0.98


DEFEAT_TEXT_PATTERNS = (
    re.compile(r"\bout\s+of\s+us(?:e?able)?\b", re.IGNORECASE),
    re.compile(r"\b(?:blacked|whited)\s+out\b", re.IGNORECASE),
    re.compile(r"\b(?:you|player)\s+(?:black(?:ed)?|whit(?:ed)?)\b", re.IGNORECASE),
    re.compile(r"\bscurried\s+(?:to|back)\b", re.IGNORECASE),
)

PLAYER_FAINT_PATTERN = re.compile(
    r"(?:^|[\n.!])\s*(?!enemy\b|wild\b|foe\b)[A-Za-z0-9_.'♀♂-]+\s+fainted\b",
    re.IGNORECASE,
)


def is_battle_defeat_text(texts: Iterable[str]) -> bool:
    """Return True if any dialogue string indicates the player lost all usable Pokemon and blacked/whited out."""
    combined = " ".join(str(t) for t in (texts or ()) if t)
    if not combined:
        return False
    return any(pattern.search(combined) is not None for pattern in DEFEAT_TEXT_PATTERNS)


def has_player_pokemon_fainted_text(texts: Iterable[str]) -> bool:
    """Return True if any dialogue string reports one of the player's own Pokemon fainting (excluding Enemy/Wild/Foe)."""
    for text in (texts or ()):
        if text and PLAYER_FAINT_PATTERN.search(str(text)) is not None:
            return True
    return False
