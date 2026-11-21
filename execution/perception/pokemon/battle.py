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
    return 1 <= n_border_cols <= MAX_BORDER_COLS


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
