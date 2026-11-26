import re
from typing import Iterable, List, Optional, Tuple

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
    if not (5 <= len(nz_rows) <= 7) or nz_rows[-1] - nz_rows[0] + 1 != len(nz_rows):
        return False
    rmax = max(row_sums)
    if rmax < 4 or row_sums.count(rmax) != 1 or (row_sums[3] != rmax and row_sums[4] != rmax):
        return False
    if row_sums[0] > 1 and row_sums[7] > 1:
        return False
    for k in range(len(nz_rows) // 2):
        if abs(row_sums[nz_rows[k]] - row_sums[nz_rows[-1 - k]]) > 1:
            return False
    r_vals = [row_sums[r] for r in nz_rows]
    mid = len(r_vals) // 2
    if any(r_vals[i] >= r_vals[i + 1] for i in range(mid)):
        return False
    if any(r_vals[i] <= r_vals[i + 1] for i in range(mid, len(r_vals) - 1)):
        return False
    left_col = nz_cols[0]
    for r in nz_rows:
        cols = [c for c in range(TILE_SIZE) if dark[r, c]]
        if cols[0] != left_col or cols[-1] - cols[0] + 1 != len(cols):
            return False
    return True


def has_menu_cursor(frame: np.ndarray) -> bool:
    """Return True if any 8x8 grid tile on screen contains a right-pointing menu cursor (or active Pack item list)."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    max_ty = min(GRID_ROWS, img.shape[0] // TILE_SIZE)
    max_tx = min(GRID_COLS, img.shape[1] // TILE_SIZE)
    for ty in range(max_ty):
        y0 = ty * TILE_SIZE
        for tx in range(max_tx):
            x0 = tx * TILE_SIZE
            if is_cursor_tile(img[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]):
                return True
    return is_pack_item_list_screen(img)


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


def is_battle_command_menu(frame: np.ndarray) -> bool:
    """Return True when the 2x2 battle command menu (FIGHT / PKMN / ITEM|PACK / RUN) is open in the bottom box."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 8 * TILE_SIZE:9 * TILE_SIZE])
               for ty in range(13, 17)):
        return False
    gutter = img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM, TILE_SIZE:8 * TILE_SIZE]
    if gutter.size == 0 or float((gutter >= DIALOGUE_BOX_WHITE).mean()) < 0.85:
        return False
    return any(
        is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
        for ty in range(13, 17)
        for tx in range(9, 19)
    )


def is_battle_decision_screen(frame: np.ndarray) -> bool:
    """Return True when the battle screen displays a command menu, move list, or cursor choice box."""
    return has_battle_menu_box(frame) or has_menu_cursor(frame)


BATTLE_FRAME_DIFF_TOLERANCE = 32


def is_same_battle_frame(earlier: np.ndarray, later: np.ndarray) -> bool:
    """Return True if two battle frames are identical (within minor DMG/CGB fade tolerance) outside the bottom-right scroll-arrow corner."""
    if earlier.shape != later.shape:
        return False
    if np.array_equal(earlier, later):
        return True
    a = _grey_frame(earlier).astype(int)
    b = _grey_frame(later).astype(int)
    differ = np.abs(a - b) > BATTLE_FRAME_DIFF_TOLERANCE
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
CUTSCENE_TILE_MIN_PIXELS = 10
CUTSCENE_MAX_TILE_SHIFT = 2
CUTSCENE_DIFF_MIN_PIXELS = 80
CUTSCENE_PLAYER_SPIN_MIN_PIXELS = 40
FADE_UPPER_BOTTOM = 88
FADE_MAX_RANGE = 16


def is_blank_or_fade_frame(frame: np.ndarray) -> bool:
    """Return True if the upper overworld viewport is a uniform white, grey, or black fade/blank transition frame."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < FADE_UPPER_BOTTOM:
        return False
    upper = img[:FADE_UPPER_BOTTOM, :]
    return int(upper.max()) - int(upper.min()) <= FADE_MAX_RANGE


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


def is_cutscene_frame_changing(
    earlier: np.ndarray, later: np.ndarray, ignore_player_box: bool = False
) -> bool:
    """Return True if sprite movement, emote animation, or box transitions changed between frames."""
    if earlier is None or later is None or earlier.shape != later.shape:
        return False
    if np.array_equal(earlier, later):
        return False
    a = _grey_frame(earlier).astype(int)
    b = _grey_frame(later).astype(int)
    diff = np.abs(a - b) > CUTSCENE_DIFF_THRESHOLD
    diff[96:, 104:] = False
    if ignore_player_box:
        diff[56:80, 64:80] = False
    raw_sum = int(diff.sum())
    if raw_sum < CUTSCENE_DIFF_MIN_PIXELS:
        return False
    n_rows = (diff.shape[0] // TILE_SIZE) * TILE_SIZE
    n_cols = (diff.shape[1] // TILE_SIZE) * TILE_SIZE
    if n_rows == 0 or n_cols == 0:
        return raw_sum >= CUTSCENE_DIFF_MIN_PIXELS
    ty, tx = n_rows // TILE_SIZE, n_cols // TILE_SIZE
    t_min = diff[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
    for d in range(1, CUTSCENE_MAX_TILE_SHIFT + 1):
        if d < n_cols:
            diff_l = np.ones_like(diff)
            diff_l[:, d:] = np.abs(b[:, d:] - a[:, :-d]) > CUTSCENE_DIFF_THRESHOLD
            diff_l[96:, 104:] = False
            if ignore_player_box:
                diff_l[56:80, 64:80] = False
            diff_r = np.ones_like(diff)
            diff_r[:, :-d] = np.abs(b[:, :-d] - a[:, d:]) > CUTSCENE_DIFF_THRESHOLD
            diff_r[96:, 104:] = False
            if ignore_player_box:
                diff_r[56:80, 64:80] = False
            tl = diff_l[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            tr = diff_r[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            t_min = np.minimum(t_min, np.minimum(tl, tr))
        if d < n_rows:
            diff_u = np.ones_like(diff)
            diff_u[d:, :] = np.abs(b[d:, :] - a[:-d, :]) > CUTSCENE_DIFF_THRESHOLD
            diff_u[96:, 104:] = False
            if ignore_player_box:
                diff_u[56:80, 64:80] = False
            diff_d = np.ones_like(diff)
            diff_d[:-d, :] = np.abs(b[:-d, :] - a[d:, :]) > CUTSCENE_DIFF_THRESHOLD
            diff_d[96:, 104:] = False
            if ignore_player_box:
                diff_d[56:80, 64:80] = False
            tu = diff_u[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            td = diff_d[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            t_min = np.minimum(t_min, np.minimum(tu, td))
    a_tiles = a[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).swapaxes(1, 2)
    b_tiles = b[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).swapaxes(1, 2)
    pair_counts = {}
    pair_keys = {}
    for iy in range(ty):
        for ix in range(tx):
            if t_min[iy, ix] < CUTSCENE_TILE_MIN_PIXELS:
                continue
            bt = b_tiles[iy, ix]
            if int(bt.min()) == int(bt.max()):
                continue
            key = (a_tiles[iy, ix].tobytes(), bt.tobytes())
            pair_keys[(iy, ix)] = key
            pair_counts[key] = pair_counts.get(key, 0) + 1
    for (iy, ix), key in pair_keys.items():
        if pair_counts[key] >= 2:
            t_min[iy, ix] = 0
    if int(t_min[t_min >= CUTSCENE_TILE_MIN_PIXELS].sum()) >= CUTSCENE_DIFF_MIN_PIXELS:
        return True
    if not ignore_player_box:
        player_tiles = t_min[7:10, 8:10]
        if int(player_tiles[player_tiles >= CUTSCENE_TILE_MIN_PIXELS].sum()) >= CUTSCENE_PLAYER_SPIN_MIN_PIXELS:
            return True
    return False


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


def has_quantity_selector_box(frame: np.ndarray) -> bool:
    """Return True if an enclosed quantity/price selector sub-box is open above the bottom dialogue box."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    max_ty = min(GRID_ROWS, img.shape[0] // TILE_SIZE)
    max_tx = min(GRID_COLS, img.shape[1] // TILE_SIZE)
    if max_ty < GRID_ROWS or max_tx < GRID_COLS:
        return False
    tiles = [[img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
              for tx in range(max_tx)] for ty in range(max_ty)]
    h_border = [[is_horizontal_border_tile(tiles[ty][tx]) for tx in range(max_tx)] for ty in range(max_ty)]
    v_border = [[is_vertical_border_tile(tiles[ty][tx]) for tx in range(max_tx)] for ty in range(max_ty)]
    for top_ty in range(6, 11):
        for bot_ty in range(top_ty + 2, min(top_ty + 5, 13)):
            for left_tx in range(5, 14):
                for right_tx in range(max(left_tx + 5, 15), 20):
                    check_end = min(right_tx, 13)
                    if check_end <= left_tx + 2:
                        continue
                    if not all(h_border[top_ty][tx] and h_border[bot_ty][tx]
                               for tx in range(left_tx + 1, check_end)):
                        continue
                    if not all(v_border[ty][left_tx] for ty in range(top_ty + 1, bot_ty)):
                        continue
                    y0, y1 = (top_ty + 1) * TILE_SIZE, bot_ty * TILE_SIZE
                    x0, x1 = (left_tx + 1) * TILE_SIZE, check_end * TILE_SIZE
                    inner = img[y0:y1, x0:x1]
                    if inner.size == 0 or float((inner >= LIGHT_BACKGROUND).mean()) < 0.50:
                        continue
                    if int((inner < DARK_INK).sum()) < 15:
                        continue
                    return True
    return False


def has_emote_bubble(frame: np.ndarray) -> bool:
    """Return True if an overworld speech/exclamation ('!') emote bubble is drawn above a sprite."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    h, w = img.shape[:2]
    if h < 120 or w < 160:
        return False
    max_y = min(h - 9, 96)
    dark = img < DARK_INK
    light = img >= LIGHT_BACKGROUND
    for x0 in range(0, w - 15, 4):
        bot_rows = dark[8:max_y + 8, x0 + 3:x0 + 13].all(axis=1)
        for y_core in np.flatnonzero(bot_rows):
            y_core = int(y_core)
            y_bot = y_core + 8
            if not (dark[y_core:y_core + 7, x0 + 1].all() and dark[y_core:y_core + 7, x0 + 14].all()):
                continue
            if not (dark[y_bot - 1, x0 + 2] and dark[y_bot - 1, x0 + 13] and light[y_bot - 1, x0 + 3:x0 + 13].all()):
                continue
            if float(light[y_core:y_core + 7, x0 + 2:x0 + 7].mean()) < 0.85:
                continue
            if float(light[y_core:y_core + 7, x0 + 9:x0 + 14].mean()) < 0.85:
                continue
            if not (int(dark[y_core:y_core + 7, x0 + 6:x0 + 10].sum()) >= 6
                    and int(light[y_core:y_core + 7, x0 + 6:x0 + 10].sum()) >= 6):
                continue
            if y_core >= 3 and not dark[y_core - 3, x0 + 3:x0 + 13].all():
                continue
            return True
    return False


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


def is_party_menu_screen(frame: np.ndarray) -> bool:
    """Return True if the upper screen shows the Generation 1 or Generation 2 party menu layout."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    dark = img < DARK_INK
    light = img >= LIGHT_BACKGROUND
    if (
        float(dark[22, 88:156].mean()) >= 0.90
        and float(dark[17, 88:104].mean()) >= 0.85
        and float(dark[17:23, 152:156].mean()) >= 0.85
        and float(light[16, 88:156].mean()) >= 0.95
        and float(light[23, 88:156].mean()) >= 0.95
        and int(dark[8:16, 8:80].sum()) >= 40
    ):
        return True
    if (
        float(dark[10, 48:96].mean()) >= 0.90
        and float(dark[13, 48:96].mean()) >= 0.90
        and float(light[8:10, 48:96].mean()) >= 0.95
        and float(light[14:16, 48:96].mean()) >= 0.95
        and int(dark[10:14, 32:48].sum()) >= 18
        and int(dark[0:8, 8:96].sum()) >= 40
    ):
        return True
    return False


def is_pack_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows the Generation 2 Pack screen (with or without an open sub-box)."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    top_bar = img[:TILE_SIZE, :]
    if float((top_bar < DARK_INK).mean()) < 0.65 or int((top_bar >= LIGHT_BACKGROUND).sum()) < 40:
        return False
    sidebar = img[TILE_SIZE:BATTLE_UPPER_BOTTOM, :5 * TILE_SIZE]
    label_band = img[58:76, 2:38]
    if float((sidebar < LIGHT_BACKGROUND).mean()) < 0.65 or float((label_band < DARK_INK).mean()) < 0.60:
        return False
    item_area = img[TILE_SIZE:BATTLE_UPPER_BOTTOM, 5 * TILE_SIZE:GRID_COLS * TILE_SIZE]
    if float((item_area >= LIGHT_BACKGROUND).mean()) < 0.70 or int((item_area < DARK_INK).sum()) < 25:
        return False
    return True


def _max_pack_action_box_border_tiles(img: np.ndarray) -> int:
    """Return the maximum vertical border tile count across the Pack action sub-box columns."""
    return max(
        sum(
            is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
            for ty in range(3, 12)
        )
        for tx in range(10, 16)
    )


def is_pack_item_list_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows the Generation 2 Pack item list without an open action/quantity sub-box."""
    if not is_pack_screen(frame):
        return False
    img = _grey_frame(frame)
    if _max_pack_action_box_border_tiles(img) >= 4:
        return False
    if has_quantity_selector_box(img):
        return False
    return True


def is_overworld_pack_menu(frame: np.ndarray) -> bool:
    """Return True if a Generation 2 Pack screen has an overworld-only multi-row action box (3+ options) or quantity selector open."""
    if not is_pack_screen(frame):
        return False
    img = _grey_frame(frame)
    if has_quantity_selector_box(img):
        return True
    return _max_pack_action_box_border_tiles(img) >= 6


def is_start_menu_screen(frame: np.ndarray) -> bool:
    """Return True if the right-hand overworld Start menu box is open on screen."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if has_bottom_dialogue_box(img) or is_party_menu_screen(img) or is_pack_screen(img):
        return False
    for left_tx in (9, 10):
        x0 = left_tx * TILE_SIZE
        n_left_v = sum(
            is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, x0:x0 + TILE_SIZE])
            for ty in range(1, 15)
        )
        n_right_v = sum(
            is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 19 * TILE_SIZE:20 * TILE_SIZE])
            for ty in range(1, 15)
        )
        if n_left_v >= 10 and n_right_v >= 10:
            inner = img[TILE_SIZE:14 * TILE_SIZE, (left_tx + 1) * TILE_SIZE:19 * TILE_SIZE]
            if float((inner >= LIGHT_BACKGROUND).mean()) >= 0.70:
                return True
    return False


def is_battle_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows a battle backdrop above a bottom battle box."""
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if (has_top_dialogue_box(img)
            or has_modal_preview_box(img)
            or is_party_menu_screen(img)
            or is_pack_screen(img)
            or is_start_menu_screen(img)):
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


def is_evolution_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows a full-screen evolution stage above a non-empty bottom dialogue box."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if (has_battle_menu_box(img)
            or has_menu_cursor(img)
            or has_top_dialogue_box(img)
            or has_modal_preview_box(img)):
        return False
    box_inner = img[BOTTOM_BOX_INNER_TOP:BOTTOM_BOX_INNER_BOTTOM, BOTTOM_BOX_INNER_LEFT:112]
    if int((box_inner < DARK_INK).sum()) < 20:
        return False
    left_margin = img[:94, 8:44]
    right_margin = img[:94, 116:152]
    lower_strip = img[76:94, 8:152]
    if (int((left_margin < LIGHT_BACKGROUND).sum()) == 0
            and int((right_margin < LIGHT_BACKGROUND).sum()) == 0
            and int((lower_strip < LIGHT_BACKGROUND).sum()) == 0):
        return True
    if (float((left_margin < 40).mean()) >= 0.90
            and float((right_margin < 40).mean()) >= 0.90
            and float((lower_strip < 40).mean()) >= 0.90
            and int((img[16:76, 48:112] >= 80).sum()) >= 50):
        return True
    return False


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
    if is_pack_item_list_screen(img):
        if not bool((img[2 * TILE_SIZE:3 * TILE_SIZE, 8 * TILE_SIZE:14 * TILE_SIZE] < DARK_INK).any()):
            return False
        below_pack = img[3 * TILE_SIZE:11 * TILE_SIZE, 5 * TILE_SIZE:20 * TILE_SIZE]
        return int((below_pack < DARK_INK).sum()) <= 5 and float((below_pack >= DIALOGUE_BOX_WHITE).mean()) >= 0.98
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


_ROW_BIT_WEIGHTS = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

HUD_SLASH_TEMPLATES = {
    (0, 2, 4, 8, 16, 32, 64, 128),
    (2, 4, 8, 16, 32, 64, 128, 0),
}

HUD_DIGIT_TEMPLATES = {
    (0, 56, 76, 198, 198, 100, 56, 0): 0,
    (0, 24, 56, 24, 24, 24, 126, 0): 1,
    (0, 124, 198, 14, 120, 224, 254, 0): 2,
    (0, 124, 198, 14, 124, 224, 254, 0): 2,
    (0, 126, 12, 56, 6, 198, 124, 0): 3,
    (0, 28, 60, 108, 204, 254, 12, 0): 4,
    (0, 252, 128, 252, 6, 198, 124, 0): 5,
    (0, 252, 192, 252, 6, 198, 124, 0): 5,
    (0, 124, 192, 252, 198, 198, 124, 0): 6,
    (0, 254, 198, 12, 24, 48, 48, 0): 7,
    (0, 124, 198, 124, 198, 198, 124, 0): 8,
    (0, 124, 198, 198, 126, 6, 124, 0): 9,
}


def _tile_row_signature(tile_2d: np.ndarray) -> Tuple[int, ...]:
    dark = (tile_2d < DARK_INK).astype(np.uint8)
    return tuple(int(v) for v in (dark @ _ROW_BIT_WEIGHTS))


def _read_tile_hp_fraction(img: np.ndarray, ty: int, slash_tx: int) -> Optional[Tuple[int, int]]:
    y0 = ty * TILE_SIZE
    if img.shape[0] < y0 + TILE_SIZE or img.shape[1] < (slash_tx + 4) * TILE_SIZE:
        return None
    slash_sig = _tile_row_signature(img[y0:y0 + TILE_SIZE, slash_tx * TILE_SIZE:(slash_tx + 1) * TILE_SIZE])
    if slash_sig not in HUD_SLASH_TEMPLATES:
        return None
    cur_digits: List[str] = []
    max_digits: List[str] = []
    for tx, target in (
        (slash_tx - 3, cur_digits),
        (slash_tx - 2, cur_digits),
        (slash_tx - 1, cur_digits),
        (slash_tx + 1, max_digits),
        (slash_tx + 2, max_digits),
        (slash_tx + 3, max_digits),
    ):
        tile = img[y0:y0 + TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
        if float((tile >= LIGHT_BACKGROUND).mean()) < 0.50:
            return None
        sig = _tile_row_signature(tile)
        if not any(sig):
            if target:
                return None
            continue
        digit = HUD_DIGIT_TEMPLATES.get(sig)
        if digit is None:
            return None
        target.append(str(digit))
    if not cur_digits or not max_digits:
        return None
    cur_hp, max_hp = int("".join(cur_digits)), int("".join(max_digits))
    if 0 <= cur_hp <= max_hp and 10 <= max_hp <= 999:
        return (cur_hp, max_hp)
    return None


def read_battle_hud_hp(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cur_hp, max_hp) read from the player's battle HUD at tile row 10, or None if not visible."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return None
    if is_party_menu_screen(img) or is_pack_screen(img):
        return None
    return _read_tile_hp_fraction(img, ty=10, slash_tx=14)


def read_party_menu_hp(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cur_hp, max_hp) read from the selected (or lead) Pokemon on the party menu screen, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if not is_party_menu_screen(img):
        return None
    for slot_ty in (1, 3, 5, 7, 9, 11):
        for cursor_ty in (slot_ty - 1, slot_ty):
            if 0 <= cursor_ty < 12 and is_cursor_tile(
                img[cursor_ty * TILE_SIZE:(cursor_ty + 1) * TILE_SIZE, :TILE_SIZE]
            ):
                hp = _read_tile_hp_fraction(img, ty=slot_ty, slash_tx=16)
                if hp is not None:
                    return hp
    return _read_tile_hp_fraction(img, ty=1, slash_tx=16)
