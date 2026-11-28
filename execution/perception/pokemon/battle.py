import re
from typing import Dict, Iterable, List, Optional, Tuple

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
CUTSCENE_TILE_MIN_PIXELS = 7
CUTSCENE_MAX_TILE_SHIFT = 2
CUTSCENE_DIFF_MIN_PIXELS = 45
CUTSCENE_PLAYER_SPIN_MIN_PIXELS = 40
CUTSCENE_MEAN_SHIFT_MAX = 18.0
CUTSCENE_SCREEN_WIDE_MIN_TILES = 20
CUTSCENE_SCREEN_WIDE_MIN_PIXELS = 450
FADE_UPPER_BOTTOM = 88
FADE_MAX_RANGE = 16
FADE_SPRITE_MAX_P95 = 230
WAVY_WIPE_MAX_SHIFT = 48
WAVY_WIPE_ROW_TOLERANCE = 24
WAVY_WIPE_ROW_MATCH_FRACTION = 0.78
WAVY_WIPE_MIN_MATCHED_ROWS_FRACTION = 0.65
WAVY_WIPE_MAX_UNIFORM_ROWS_FRACTION = 0.45
WAVY_WIPE_MIN_SPREAD_ROWS_FRACTION = 0.35


def is_blank_or_fade_frame(frame: np.ndarray) -> bool:
    """Return True if the upper overworld viewport is a uniform white, grey, or black fade/blank transition frame."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < FADE_UPPER_BOTTOM:
        return False
    upper = img[:FADE_UPPER_BOTTOM, :]
    if int(upper.max()) - int(upper.min()) <= FADE_MAX_RANGE:
        return True
    p5, p95 = np.percentile(upper, (5, 95))
    return bool((p95 - p5) <= FADE_MAX_RANGE and p95 < FADE_SPRITE_MAX_P95)


def is_wavy_scanline_wipe_against(ref_frame: np.ndarray, cur_frame: np.ndarray) -> bool:
    """Return True if cur_frame is a horizontal wavy scanline-shifted battle-wipe distortion of ref_frame."""
    if ref_frame is None or cur_frame is None:
        return False
    ref_full = _grey_frame(ref_frame).astype(np.int16)
    cur_full = _grey_frame(cur_frame).astype(np.int16)
    if ref_full.shape != cur_full.shape or ref_full.shape[0] < FADE_UPPER_BOTTOM:
        return False
    H = FADE_UPPER_BOTTOM
    cur = cur_full[:H, :]
    n_shifts = 2 * WAVY_WIPE_MAX_SHIFT + 1
    for vy in (0, -16, 16):
        if vy == 0:
            ref = ref_full[:H, :]
            c_sub = cur
            y_len = H
        elif vy > 0:
            ref = ref_full[vy:H, :]
            c_sub = cur[:H - vy, :]
            y_len = H - vy
        else:
            va = -vy
            ref = ref_full[:H - va, :]
            c_sub = cur[va:H, :]
            y_len = H - va
        match_mat = np.zeros((n_shifts, y_len), dtype=bool)
        for idx_s, s in enumerate(range(-WAVY_WIPE_MAX_SHIFT, WAVY_WIPE_MAX_SHIFT + 1)):
            if s == 0:
                m = np.mean(np.abs(c_sub - ref) <= WAVY_WIPE_ROW_TOLERANCE, axis=1)
            elif s > 0:
                m = np.mean(np.abs(c_sub[:, s:] - ref[:, :-s]) <= WAVY_WIPE_ROW_TOLERANCE, axis=1)
            else:
                sa = -s
                m = np.mean(np.abs(c_sub[:, :-sa] - ref[:, sa:]) <= WAVY_WIPE_ROW_TOLERANCE, axis=1)
            match_mat[idx_s] = m >= WAVY_WIPE_ROW_MATCH_FRACTION
        best_uniform = int(match_mat.sum(axis=1).max())
        per_row_matched = int(match_mat.any(axis=0).sum())
        if (
            per_row_matched >= int(WAVY_WIPE_MIN_MATCHED_ROWS_FRACTION * y_len)
            and best_uniform <= int(WAVY_WIPE_MAX_UNIFORM_ROWS_FRACTION * y_len)
            and (per_row_matched - best_uniform) >= int(WAVY_WIPE_MIN_SPREAD_ROWS_FRACTION * y_len)
        ):
            return True
    return False


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


def _evaluate_cutscene_frame_change(
    earlier: np.ndarray, later: np.ndarray, ignore_player_box: bool = False
) -> Tuple[bool, bool]:
    """Return (is_changing, is_screen_wide) between two overworld frames."""
    if earlier is None or later is None or earlier.shape != later.shape:
        return False, False
    if np.array_equal(earlier, later):
        return False, False
    a = _grey_frame(earlier).astype(int)
    b = _grey_frame(later).astype(int)
    diff = np.abs(a - b) > CUTSCENE_DIFF_THRESHOLD
    diff[96:, 104:] = False
    if ignore_player_box:
        diff[56:80, 64:80] = False
    raw_sum = int(diff.sum())
    if raw_sum < CUTSCENE_DIFF_MIN_PIXELS:
        return False, False
    if is_wavy_scanline_wipe_against(earlier, later):
        return True, True
    n_rows = (diff.shape[0] // TILE_SIZE) * TILE_SIZE
    n_cols = (diff.shape[1] // TILE_SIZE) * TILE_SIZE
    if n_rows == 0 or n_cols == 0:
        changing = raw_sum >= CUTSCENE_DIFF_MIN_PIXELS
        return changing, raw_sum >= CUTSCENE_SCREEN_WIDE_MIN_PIXELS
    ty, tx = n_rows // TILE_SIZE, n_cols // TILE_SIZE

    a_tiles = a[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).swapaxes(1, 2)
    b_tiles = b[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).swapaxes(1, 2)

    pair_counts: Dict[Tuple[bytes, bytes], int] = {}
    pair_keys: Dict[Tuple[int, int], Tuple[bytes, bytes]] = {}
    roll_tiles = set()
    roll_shade_pairs = set()
    n_two_shade_roll_tiles = 0
    for iy in range(ty):
        for ix in range(tx):
            at = a_tiles[iy, ix]
            bt = b_tiles[iy, ix]
            tile_diff = np.abs(at - bt) > CUTSCENE_DIFF_THRESHOLD
            if not np.any(tile_diff):
                continue
            if (
                int(at.max()) - int(at.min()) <= CUTSCENE_DIFF_THRESHOLD
                or int(bt.max()) - int(bt.min()) <= CUTSCENE_DIFF_THRESHOLD
                or abs(float(at.mean()) - float(bt.mean())) > CUTSCENE_MEAN_SHIFT_MAX
            ):
                continue
            for s in range(1, TILE_SIZE):
                if not np.any(np.abs(np.roll(at, s, axis=1) - bt) > CUTSCENE_DIFF_THRESHOLD):
                    roll_tiles.add((iy, ix))
                    if len(set((at // 32).flatten().tolist())) <= 2 and len(set((bt // 32).flatten().tolist())) <= 2:
                        n_two_shade_roll_tiles += 1
                        for sa, sb in zip((at[tile_diff] // 32).tolist(), (bt[tile_diff] // 32).tolist()):
                            roll_shade_pairs.add((sa, sb))
                    break
            key = ((at // 16).tobytes(), (bt // 16).tobytes())
            pair_keys[(iy, ix)] = key
            pair_counts[key] = pair_counts.get(key, 0) + 1

    bg_anim_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    seen_bg_keys = set()
    for (iy, ix), key in pair_keys.items():
        if pair_counts[key] >= 2 or (iy, ix) in roll_tiles:
            if key not in seen_bg_keys:
                seen_bg_keys.add(key)
                bg_anim_pairs.append((a_tiles[iy, ix], b_tiles[iy, ix]))

    bg_mask = np.zeros_like(diff, dtype=bool)
    if bg_anim_pairs:
        use_roll_shades = n_two_shade_roll_tiles >= 2 and bool(roll_shade_pairs)
        bg_mask_tiles = np.zeros((ty, tx, TILE_SIZE, TILE_SIZE), dtype=bool)
        for iy in range(ty):
            for ix in range(tx):
                if not np.any(np.abs(a_tiles[iy, ix] - b_tiles[iy, ix]) > CUTSCENE_DIFF_THRESHOLD):
                    continue
                key = pair_keys.get((iy, ix))
                if (key is not None and pair_counts.get(key, 0) >= 2) or (iy, ix) in roll_tiles:
                    bg_mask_tiles[iy, ix] = True
                    continue
                at = a_tiles[iy, ix]
                bt = b_tiles[iy, ix]
                if (
                    int(at.max()) - int(at.min()) <= CUTSCENE_DIFF_THRESHOLD
                    or int(bt.max()) - int(bt.min()) <= CUTSCENE_DIFF_THRESHOLD
                    or abs(float(at.mean()) - float(bt.mean())) > CUTSCENE_MEAN_SHIFT_MAX
                ):
                    continue
                bg_pixel_match = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
                for a_bg, b_bg in bg_anim_pairs:
                    bg_changed = np.abs(a_bg - b_bg) > CUTSCENE_DIFF_THRESHOLD
                    match_rc = (
                        bg_changed
                        & (np.abs(at - a_bg) <= CUTSCENE_DIFF_THRESHOLD)
                        & (np.abs(bt - b_bg) <= CUTSCENE_DIFF_THRESHOLD)
                    )
                    bg_pixel_match |= match_rc
                if use_roll_shades:
                    qa = at // 32
                    qb = bt // 32
                    for sa, sb in roll_shade_pairs:
                        bg_pixel_match |= (qa == sa) & (qb == sb)
                bg_mask_tiles[iy, ix] = bg_pixel_match
        bg_mask[:n_rows, :n_cols] = bg_mask_tiles.swapaxes(1, 2).reshape(n_rows, n_cols)
        diff = diff & ~bg_mask

    if int(diff.sum()) < CUTSCENE_DIFF_MIN_PIXELS:
        return False, False

    t_min = diff[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
    for d in range(1, CUTSCENE_MAX_TILE_SHIFT + 1):
        if d < n_cols:
            diff_l = np.ones_like(diff)
            diff_l[:, d:] = np.abs(b[:, d:] - a[:, :-d]) > CUTSCENE_DIFF_THRESHOLD
            diff_l[96:, 104:] = False
            if ignore_player_box:
                diff_l[56:80, 64:80] = False
            diff_l &= ~bg_mask
            diff_r = np.ones_like(diff)
            diff_r[:, :-d] = np.abs(b[:, :-d] - a[:, d:]) > CUTSCENE_DIFF_THRESHOLD
            diff_r[96:, 104:] = False
            if ignore_player_box:
                diff_r[56:80, 64:80] = False
            diff_r &= ~bg_mask
            tl = diff_l[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            tr = diff_r[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            t_min = np.minimum(t_min, np.minimum(tl, tr))
        if d < n_rows:
            diff_u = np.ones_like(diff)
            diff_u[d:, :] = np.abs(b[d:, :] - a[:-d, :]) > CUTSCENE_DIFF_THRESHOLD
            diff_u[96:, 104:] = False
            if ignore_player_box:
                diff_u[56:80, 64:80] = False
            diff_u &= ~bg_mask
            diff_d = np.ones_like(diff)
            diff_d[:-d, :] = np.abs(b[:-d, :] - a[d:, :]) > CUTSCENE_DIFF_THRESHOLD
            diff_d[96:, 104:] = False
            if ignore_player_box:
                diff_d[56:80, 64:80] = False
            diff_d &= ~bg_mask
            tu = diff_u[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            td = diff_d[:n_rows, :n_cols].reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
            t_min = np.minimum(t_min, np.minimum(tu, td))

    sig_tiles = t_min >= CUTSCENE_TILE_MIN_PIXELS
    n_sig_tiles = int(sig_tiles.sum())
    sig_sum = int(t_min[sig_tiles].sum())
    changing = sig_sum >= CUTSCENE_DIFF_MIN_PIXELS
    if not changing and not ignore_player_box:
        player_tiles = t_min[7:10, 8:10]
        if int(player_tiles[player_tiles >= CUTSCENE_TILE_MIN_PIXELS].sum()) >= CUTSCENE_PLAYER_SPIN_MIN_PIXELS:
            changing = True
    screen_wide = changing and (
        n_sig_tiles >= CUTSCENE_SCREEN_WIDE_MIN_TILES
        or sig_sum >= CUTSCENE_SCREEN_WIDE_MIN_PIXELS
    )
    return changing, screen_wide


def is_cutscene_frame_changing(
    earlier: np.ndarray, later: np.ndarray, ignore_player_box: bool = False
) -> bool:
    """Return True if sprite movement, emote animation, or box transitions changed between frames."""
    changing, _ = _evaluate_cutscene_frame_change(earlier, later, ignore_player_box=ignore_player_box)
    return changing


def is_screen_wide_cutscene_change(
    earlier: np.ndarray, later: np.ndarray, ignore_player_box: bool = False
) -> bool:
    """Return True if a screen-wide palette flash, battle wipe, or camera scroll occurred between frames."""
    _, screen_wide = _evaluate_cutscene_frame_change(earlier, later, ignore_player_box=ignore_player_box)
    return screen_wide


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


LEVEL_GLYPH_SIGNATURE = (0, 0, 24, 88, 24, 88, 30, 0)
HOLLOW_CURSOR_SIGNATURES = {
    (96, 80, 72, 68, 72, 80, 96, 0),
    (0, 96, 80, 72, 68, 72, 80, 96),
}


def is_party_cursor_tile(raw_tile: np.ndarray) -> bool:
    """Return True if an 8x8 tile shows either a solid or hollow right-pointing party cursor."""
    if is_cursor_tile(raw_tile):
        return True
    if raw_tile is None or raw_tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
        return False
    tile_2d = _grey_frame(raw_tile)
    if float((tile_2d >= LIGHT_BACKGROUND).mean()) < 0.50:
        return False
    return _tile_row_signature(tile_2d) in HOLLOW_CURSOR_SIGNATURES


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
        and int(dark[8:16, 8:80].sum()) >= 20
    ):
        return True
    if (
        float(dark[10, 48:96].mean()) >= 0.90
        and float(dark[13, 48:96].mean()) >= 0.90
        and float(light[8:10, 48:96].mean()) >= 0.95
        and float(light[14:16, 48:96].mean()) >= 0.95
        and int(dark[10:14, 32:48].sum()) >= 18
        and (
            int(dark[0:8, 8:96].sum()) >= 20
            or _read_tile_hp_fraction(img, ty=1, slash_tx=16) is not None
            or float(light[0:8, 0:GRID_COLS * TILE_SIZE].mean()) >= 0.95
        )
    ):
        return True
    if (
        float(light[0:TILE_SIZE, 3 * TILE_SIZE:GRID_COLS * TILE_SIZE].mean()) >= 0.95
        and len(read_tile_text(img, 1, 3, 13)) >= 2
        and any(is_party_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 0:TILE_SIZE]) for ty in (1, 3, 5, 7, 9, 11))
    ):
        return True
    if (
        len(read_tile_text(img, 0, 3, 13)) >= 2
        and any(is_party_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 0:TILE_SIZE]) for ty in (1, 3, 5, 7, 9, 11))
    ):
        return True
    return False


def is_party_menu_selection_screen(frame: np.ndarray) -> bool:
    """Return True if the party menu screen is waiting for a Pokemon target or action selection rather than displaying an effect dialogue box."""
    if not is_party_menu_screen(frame):
        return False
    img = _grey_frame(frame)
    if has_menu_cursor(img):
        return True
    return not has_bottom_dialogue_box(img)


def is_pack_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows the Generation 2 Pack screen (with or without an open sub-box or transition frame)."""
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
    if float((item_area >= LIGHT_BACKGROUND).mean()) < 0.70:
        return False
    return True


def pack_action_box_height(frame: np.ndarray) -> int:
    """Return the vertical tile height of an enclosed Pack action sub-box, or 0 if none is open."""
    if frame is None:
        return 0
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return 0
    max_h = 0
    for left_tx in range(10, 16):
        span_end = min(19, left_tx + 5)
        if span_end <= left_tx + 2:
            continue
        for top_ty in range(3, 10):
            if not all(
                is_horizontal_border_tile(img[top_ty * TILE_SIZE:(top_ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
                for tx in range(left_tx + 1, span_end)
            ):
                continue
            for bot_ty in range(top_ty + 3, 12):
                if not all(
                    is_horizontal_border_tile(img[bot_ty * TILE_SIZE:(bot_ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
                    for tx in range(left_tx + 1, span_end)
                ):
                    continue
                if all(
                    is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, left_tx * TILE_SIZE:(left_tx + 1) * TILE_SIZE])
                    for ty in range(top_ty + 1, bot_ty)
                ):
                    max_h = max(max_h, bot_ty - top_ty + 1)
    return max_h


def is_pack_item_list_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows the Generation 2 Pack item list without an open action/quantity sub-box or TM/HM startup dialogue."""
    if not is_pack_screen(frame):
        return False
    img = _grey_frame(frame)
    if pack_action_box_height(img) >= 4:
        return False
    if has_quantity_selector_box(img):
        return False
    if is_tm_hm_startup_dialogue_frame(img):
        return False
    return True


def is_pack_action_subbox_screen(frame: np.ndarray) -> bool:
    """Return True if a Generation 2 Pack screen has a 2-option in-battle USE/QUIT action sub-box open."""
    if not is_pack_screen(frame):
        return False
    img = _grey_frame(frame)
    if has_quantity_selector_box(img):
        return False
    return 4 <= pack_action_box_height(img) <= 5


def is_overworld_pack_menu(frame: np.ndarray) -> bool:
    """Return True if a Generation 2 Pack screen has an overworld-only multi-row action box (3+ options), quantity selector, or TM/HM pocket open."""
    if not is_pack_screen(frame):
        return False
    img = _grey_frame(frame)
    if has_quantity_selector_box(img) or is_tm_hm_pack_pocket(img) or is_tm_hm_startup_dialogue_frame(img):
        return True
    return pack_action_box_height(img) >= 6


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


def has_right_aligned_ui_box(frame: np.ndarray) -> bool:
    """Return True if a bordered menu/choice box is anchored along the right edge (col 19) in the upper screen."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < 12 * TILE_SIZE or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    right_v19 = sum(
        1 for ty in range(1, 12)
        if is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 19 * TILE_SIZE:20 * TILE_SIZE])
    )
    if right_v19 < 3:
        return False
    return any(
        all(
            is_horizontal_border_tile(img[top_ty * TILE_SIZE:(top_ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
            for tx in range(15, 19)
        )
        for top_ty in range(0, 10)
    )


def is_overworld_menu_or_ui_screen(frame: np.ndarray) -> bool:
    """Return True if the screen displays an overworld or full-screen menu/UI layout (party, Pack, Start menu, top box, or modal preview)."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    return (
        has_top_dialogue_box(img)
        or has_modal_preview_box(img)
        or is_party_menu_screen(img)
        or is_pack_screen(img)
        or is_start_menu_screen(img)
    )


def has_positive_battle_layout(frame: np.ndarray) -> bool:
    """Return True if the upper screen and bottom box exhibit positive visual battle layout features."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if is_battle_command_menu(img) or is_move_selection_screen(img) or has_battle_menu_box(img):
        return True
    if _read_tile_hp_fraction(img, ty=10, slash_tx=14) is not None:
        return True
    dark = img < DARK_INK
    light = img >= LIGHT_BACKGROUND
    opp_frame_line = (
        float(dark[26:32, 16:88].max(axis=0).mean()) >= 0.85
        and float(light[0:16, 96:160].mean()) >= 0.75
        and float(light[0:24, 16:88].mean()) >= 0.55
    )
    plr_frame_line = (
        float(dark[90:96, 80:152].max(axis=0).mean()) >= 0.80
        and float(light[56:88, 80:152].mean()) >= 0.55
    )
    opp_hud_header = (
        len(read_tile_text(img, 0, 1, 11)) >= 2
        and float(light[0:16, 96:160].mean()) >= 0.80
        and (
            any(
                _tile_row_signature(img[8:16, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]) == LEVEL_GLYPH_SIGNATURE
                for tx in range(3, 10)
            )
            or float(dark[18:23, 24:72].max(axis=0).mean()) >= 0.65
        )
    )
    plr_level = any(
        _tile_row_signature(img[8 * TILE_SIZE:9 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]) == LEVEL_GLYPH_SIGNATURE
        for tx in range(12, 18)
    )
    if opp_frame_line or plr_frame_line or opp_hud_header or plr_level:
        return True
    if any(is_party_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 0:TILE_SIZE]) for ty in range(12)):
        return False
    if any(len(read_tile_text(img, ty, 0, 12)) >= 2 for ty in range(5)):
        return False
    if has_right_aligned_ui_box(img):
        return False
    return True


def is_battle_screen(frame: np.ndarray) -> bool:
    """Return True if the screen shows a battle backdrop and positive battle layout above a bottom battle box."""
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if is_overworld_menu_or_ui_screen(img):
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
    return has_positive_battle_layout(img)


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


HUD_LETTER_TEMPLATES: Dict[Tuple[int, ...], str] = {
    (16, 40, 40, 68, 124, 130, 130, 0): "A",
    (248, 132, 132, 252, 130, 130, 252, 0): "B",
    (60, 66, 128, 128, 128, 66, 60, 0): "C",
    (248, 132, 130, 130, 130, 132, 248, 0): "D",
    (254, 128, 128, 252, 128, 128, 254, 0): "E",
    (254, 128, 128, 252, 128, 128, 128, 0): "F",
    (60, 66, 128, 158, 130, 66, 60, 0): "G",
    (130, 130, 130, 254, 130, 130, 130, 0): "H",
    (124, 16, 16, 16, 16, 16, 124, 0): "I",
    (126, 8, 8, 8, 136, 136, 112, 0): "J",
    (132, 136, 144, 176, 200, 132, 130, 0): "K",
    (128, 128, 128, 128, 128, 128, 254, 0): "L",
    (130, 198, 170, 146, 130, 130, 130, 0): "M",
    (130, 194, 162, 146, 138, 134, 130, 0): "N",
    (56, 68, 130, 130, 130, 68, 56, 0): "O",
    (252, 130, 130, 252, 128, 128, 128, 0): "P",
    (56, 68, 130, 130, 138, 68, 58, 0): "Q",
    (252, 130, 130, 252, 136, 132, 130, 0): "R",
    (120, 132, 128, 124, 2, 130, 124, 0): "S",
    (254, 16, 16, 16, 16, 16, 16, 0): "T",
    (130, 130, 130, 130, 130, 66, 62, 0): "U",
    (130, 130, 68, 68, 40, 40, 16, 0): "V",
    (130, 146, 170, 170, 198, 198, 130, 0): "W",
    (130, 130, 146, 170, 198, 198, 130, 0): "W",
    (198, 68, 40, 16, 40, 68, 198, 0): "X",
    (130, 68, 40, 16, 16, 16, 16, 0): "Y",
    (254, 4, 8, 16, 32, 64, 254, 0): "Z",
    (48, 192, 60, 66, 126, 64, 62, 0): "E",
    (0, 0, 0, 126, 0, 0, 0, 0): "-",
    (0, 0, 0, 0, 126, 0, 0, 0): "-",
}

DIALOGUE_LOWER_TEMPLATES: Dict[Tuple[int, ...], str] = {
    (0, 0, 56, 4, 60, 68, 62, 0): "a",
    (64, 64, 64, 124, 66, 66, 124, 0): "b",
    (0, 0, 62, 64, 64, 64, 62, 0): "c",
    (2, 2, 2, 62, 66, 66, 62, 0): "d",
    (0, 0, 60, 66, 126, 64, 62, 0): "e",
    (12, 18, 16, 126, 16, 16, 16, 0): "f",
    (0, 0, 62, 66, 66, 62, 2, 124): "g",
    (64, 64, 64, 120, 68, 68, 68, 0): "h",
    (0, 16, 0, 16, 16, 16, 16, 0): "i",
    (64, 64, 70, 88, 96, 88, 70, 0): "k",
    (16, 16, 16, 16, 16, 16, 16, 0): "l",
    (0, 0, 236, 146, 146, 146, 146, 0): "m",
    (0, 0, 60, 34, 34, 34, 34, 0): "n",
    (0, 0, 60, 66, 66, 66, 60, 0): "o",
    (0, 0, 124, 66, 66, 124, 64, 64): "p",
    (0, 0, 78, 80, 96, 64, 64, 0): "r",
    (0, 0, 60, 64, 60, 2, 124, 0): "s",
    (0, 16, 124, 16, 16, 16, 12, 0): "t",
    (0, 0, 68, 68, 68, 68, 60, 0): "u",
    (0, 0, 68, 68, 68, 40, 16, 0): "v",
    (0, 0, 130, 146, 146, 170, 68, 0): "w",
    (0, 0, 68, 68, 68, 60, 4, 120): "y",
    (126, 231, 206, 24, 0, 24, 24, 0): "?",
    (24, 60, 60, 60, 24, 0, 24, 24): "!",
    (0, 0, 0, 0, 0, 24, 24, 0): ".",
}

GEN1_POKEDOLLAR_SIGNATURE = (60, 36, 60, 32, 112, 32, 112, 32)
HUD_DIGIT_TOP6_TEMPLATES: Dict[Tuple[int, ...], int] = {
    sig[:6]: digit for sig, digit in HUD_DIGIT_TEMPLATES.items()
}

STATUS_CODES = ("PSN", "BRN", "PAR", "SLP", "FRZ")

GEN1_SCROLL_ARROW_SIGNATURES = {
    (0, 254, 254, 124, 56, 16, 0, 0),
    (254, 254, 124, 56, 16, 0, 0, 0),
}


def read_tile_text(img: np.ndarray, ty: int, tx_start: int, tx_end: int) -> str:
    """Read a contiguous uppercase/digit text string from 8x8 tiles on row ty in [tx_start, tx_end)."""
    y0 = ty * TILE_SIZE
    if img.shape[0] < y0 + TILE_SIZE:
        return ""
    max_tx = min(tx_end, img.shape[1] // TILE_SIZE)
    chars: List[str] = []
    for tx in range(tx_start, max_tx):
        tile = img[y0:y0 + TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
        if float((tile >= LIGHT_BACKGROUND).mean()) < 0.45:
            break
        sig = _tile_row_signature(tile)
        if not any(sig):
            chars.append(" ")
        elif sig in HUD_LETTER_TEMPLATES:
            chars.append(HUD_LETTER_TEMPLATES[sig])
        elif sig in HUD_DIGIT_TEMPLATES:
            chars.append(str(HUD_DIGIT_TEMPLATES[sig]))
        else:
            break
    return " ".join("".join(chars).split())


def read_move_menu_pp(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cur_pp, max_pp) read from the TYPE/PP overlay box at tile row 11, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    y0 = 11 * TILE_SIZE
    if img.shape[0] < y0 + TILE_SIZE or img.shape[1] < 10 * TILE_SIZE:
        return None
    if not all(
        is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, :TILE_SIZE])
        and is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 10 * TILE_SIZE:11 * TILE_SIZE])
        for ty in range(8, 11)
    ):
        return None
    slash_sig = _tile_row_signature(img[y0:y0 + TILE_SIZE, 7 * TILE_SIZE:8 * TILE_SIZE])
    if slash_sig not in HUD_SLASH_TEMPLATES:
        return None
    cur_digits: List[str] = []
    max_digits: List[str] = []
    for tx, target in ((5, cur_digits), (6, cur_digits), (8, max_digits), (9, max_digits)):
        tile = img[y0:y0 + TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
        if float((tile >= LIGHT_BACKGROUND).mean()) < 0.45:
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
    cur_pp, max_pp = int("".join(cur_digits)), int("".join(max_digits))
    if 0 <= cur_pp <= max_pp and 1 <= max_pp <= 99:
        return (cur_pp, max_pp)
    return None


def read_move_menu_cursor_slot(frame: np.ndarray) -> Optional[int]:
    """Return the selected move slot (1..4) when the 4-move selection menu is open, or None."""
    if frame is None or not is_move_selection_screen(frame):
        return None
    img = _grey_frame(frame)
    for slot, ty in enumerate((13, 14, 15, 16), 1):
        if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 5 * TILE_SIZE:6 * TILE_SIZE]):
            return slot
    return None


def read_move_menu_names(frame: np.ndarray) -> Dict[int, str]:
    """Return a dict mapping move slot (1..4) to move name (or prefix) on an open move selection screen."""
    if frame is None or not is_move_selection_screen(frame):
        return {}
    img = _grey_frame(frame)
    names: Dict[int, str] = {}
    for slot, ty in enumerate((13, 14, 15, 16), 1):
        name = read_tile_text(img, ty, 6, 14)
        if name and name != "-":
            names[slot] = name
    return names


def read_move_menu_type(frame: np.ndarray) -> Optional[str]:
    """Return the selected move's type string (e.g. 'NORMAL', 'FIRE') from the TYPE/ overlay box at tile row 10, or None."""
    if frame is None or not is_move_selection_screen(frame):
        return None
    img = _grey_frame(frame)
    if read_move_menu_pp(img) is None:
        return None
    t_str = read_tile_text(img, 10, 1, 10)
    if t_str and len(t_str) >= 2 and t_str != "TYPE":
        return t_str
    return None


def read_battle_opponent_hud_name(frame: np.ndarray) -> Optional[str]:
    """Return the enemy Pokemon's name from the top-left battle HUD (tile row 0), or None."""
    if frame is None or not is_battle_screen(frame):
        return None
    img = _grey_frame(frame)
    for tx in (0, 1, 2, 3):
        name = read_tile_text(img, 0, tx, 12)
        if len(name) >= 2 and name not in ("LEVEL", "HP"):
            return name
    return None


def is_no_pp_dialogue_frame(frame: np.ndarray) -> bool:
    """Return True when the 'No PP left for this move!' dialogue box is displayed below a 0/max PP overlay box."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if not has_bottom_dialogue_box(img) or is_move_selection_screen(img):
        return False
    pp = read_move_menu_pp(img)
    return pp is not None and pp[0] == 0


def _read_status_code_on_row(img: np.ndarray, ty: int, tx_start: int, tx_end: int) -> Optional[str]:
    y0 = ty * TILE_SIZE
    if img.shape[0] < y0 + TILE_SIZE:
        return None
    max_tx = min(tx_end, img.shape[1] // TILE_SIZE)
    row_chars: List[str] = []
    for tx in range(tx_start, max_tx):
        tile = img[y0:y0 + TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
        if float((tile >= LIGHT_BACKGROUND).mean()) < 0.45:
            row_chars.append(" ")
            continue
        sig = _tile_row_signature(tile)
        row_chars.append(HUD_LETTER_TEMPLATES.get(sig, " "))
    joined = "".join(row_chars)
    for code in STATUS_CODES:
        if code in joined:
            return code
    return None


def read_battle_hud_status(frame: np.ndarray) -> Optional[str]:
    """Return the active Pokemon's status code ('PSN', 'BRN', 'PAR', 'SLP', 'FRZ'), '' if healthy, or None if HUD not visible."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if read_battle_hud_hp(img) is None:
        return None
    code = _read_status_code_on_row(img, ty=8, tx_start=12, tx_end=19)
    return code if code is not None else ""


def read_party_menu_status(frame: np.ndarray) -> Optional[str]:
    """Return the selected (or lead) Pokemon's status code ('PSN', 'BRN', 'PAR', 'SLP', 'FRZ'), '' if healthy, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if not is_party_menu_screen(img):
        return None
    target_slot_ty = 1
    for slot_ty in (1, 3, 5, 7, 9, 11):
        found_cursor = False
        for cursor_ty in (slot_ty - 1, slot_ty):
            if 0 <= cursor_ty < 12 and is_cursor_tile(
                img[cursor_ty * TILE_SIZE:(cursor_ty + 1) * TILE_SIZE, :TILE_SIZE]
            ):
                if _read_tile_hp_fraction(img, ty=slot_ty, slash_tx=16) is not None:
                    target_slot_ty = slot_ty
                    found_cursor = True
                    break
        if found_cursor:
            break
    if _read_tile_hp_fraction(img, ty=target_slot_ty, slash_tx=16) is None:
        return None
    for check_ty in (target_slot_ty - 1, target_slot_ty, target_slot_ty + 1):
        if 0 <= check_ty < 13:
            code = _read_status_code_on_row(img, ty=check_ty, tx_start=9, tx_end=20)
            if code is not None:
                return code
    return ""


def is_gen1_item_list_screen(frame: np.ndarray) -> bool:
    """Return True when the Generation 1 ITEM list box is open without an action/quantity sub-box."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < 13 * TILE_SIZE or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if is_party_menu_screen(img) or is_pack_screen(img):
        return False
    if has_quantity_selector_box(img):
        return False
    if not all(is_horizontal_border_tile(img[2 * TILE_SIZE:3 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
               for tx in range(5, 19)):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 4 * TILE_SIZE:5 * TILE_SIZE])
               for ty in range(3, 12)):
        return False
    if not all(is_horizontal_border_tile(img[12 * TILE_SIZE:13 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
               for tx in range(5, 13)):
        return False
    inner = img[3 * TILE_SIZE:12 * TILE_SIZE, 5 * TILE_SIZE:13 * TILE_SIZE]
    if float((inner >= LIGHT_BACKGROUND).mean()) < 0.70:
        return False
    if any(
        all(is_horizontal_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
            for tx in range(14, 19))
        for ty in (8, 9, 10)
    ):
        return False
    if any(
        _tile_row_signature(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
        == GEN1_POKEDOLLAR_SIGNATURE
        for ty in (5, 7, 9, 11)
        for tx in (13, 14, 15)
    ):
        return False
    return True


def read_dialogue_box_lines(frame: np.ndarray, tx_start: int = 1, tx_end: int = 18) -> Tuple[str, str]:
    """Read the two mixed-case text lines (tile rows 14 and 16) inside the bottom dialogue box."""
    if frame is None or not has_bottom_dialogue_box(frame):
        return ("", "")
    img = _grey_frame(frame)
    lines: List[str] = []
    max_tx = min(tx_end, img.shape[1] // TILE_SIZE)
    for ty in (14, 16):
        y0 = ty * TILE_SIZE
        if img.shape[0] < y0 + TILE_SIZE:
            lines.append("")
            continue
        chars: List[str] = []
        for tx in range(tx_start, max_tx):
            tile = img[y0:y0 + TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
            if float((tile >= LIGHT_BACKGROUND).mean()) < 0.45:
                break
            sig = _tile_row_signature(tile)
            if not any(sig):
                chars.append(" ")
            elif sig in HUD_LETTER_TEMPLATES:
                chars.append(HUD_LETTER_TEMPLATES[sig])
            elif sig in DIALOGUE_LOWER_TEMPLATES:
                chars.append(DIALOGUE_LOWER_TEMPLATES[sig])
            elif sig in HUD_DIGIT_TEMPLATES:
                chars.append(str(HUD_DIGIT_TEMPLATES[sig]))
            else:
                chars.append(" ")
        lines.append(" ".join("".join(chars).split()))
    return (lines[0], lines[1])


def is_tm_hm_startup_dialogue_frame(frame: np.ndarray) -> bool:
    """Return True when the bottom dialogue box shows a TM/HM boot or teaching confirmation prompt ('Booted up...', 'It contained...', 'Teach ... to a POKeMON?')."""
    if frame is None or not has_bottom_dialogue_box(frame) or is_party_menu_screen(frame):
        return False
    line1, line2 = read_dialogue_box_lines(frame, tx_start=1, tx_end=15)
    l1_low = line1.lower()
    l2_low = line2.lower()
    if "booted up" in l1_low or l1_low.startswith("booted ") or "booted a" in l1_low:
        return True
    if l1_low.startswith("it contain") or "contained" in l1_low:
        return True
    if l1_low.startswith("teach ") or l2_low.startswith("to a p"):
        return True
    return False


def is_tm_hm_item_name(name: Optional[str]) -> bool:
    """Return True if a bag or Pack item name is a TM or HM (e.g. 'TM34', 'HM01', '31 MUD-SLAP', 'H1 CUT')."""
    up = re.sub(r"\s*\(X\d+\)\s*$", "", (name or "").strip().upper())
    if not up or up == "CANCEL":
        return False
    return bool(re.match(r"^(?:(?:TM|HM)\s*\d|\d{2}\s+[A-Z]|H\d\s+[A-Z])", up))


def is_tm_hm_pack_pocket(frame: np.ndarray) -> bool:
    """Return True if the Generation 2 Pack screen is displaying the TM/HM pocket (with 2-character TM/HM codes at columns 5..6)."""
    if not is_pack_screen(frame):
        return False
    img = _grey_frame(frame)
    for ty in (2, 4, 6, 8, 10):
        code = read_tile_text(img, ty, 5, 7)
        if code and re.match(r"^(?:\d{2}|H\d)$", code):
            return True
    return False


def read_tm_hm_pocket_rows(frame: np.ndarray) -> Optional[List[str]]:
    """Return formatted visible rows (e.g. ['31 MUD-SLAP (x1)', 'H1 CUT', 'CANCEL']) when the Gen 2 TM/HM pocket list is open, or None."""
    if not is_tm_hm_pack_pocket(frame) or not is_pack_item_list_screen(frame):
        return None
    img = _grey_frame(frame)
    rows: List[str] = []
    for ty in (2, 4, 6, 8, 10):
        code = read_tile_text(img, ty, 5, 7)
        name = read_tile_text(img, ty, 8, 20)
        if not name:
            continue
        if name == "CANCEL" or name.startswith("CANCEL "):
            rows.append("CANCEL")
            break
        qty = read_tile_quantity(img, ty + 1, 17, 20)
        label = f"{code} {name}".strip() if code else name
        if qty is not None:
            label = f"{label} (x{qty})"
        rows.append(label)
    return rows if rows else None


def read_bag_item_list(frame: np.ndarray) -> Optional[Tuple[List[str], bool]]:
    """Return (visible_item_names, has_more_below) when a Generation 1 or Generation 2 bag item list is open, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if is_gen1_item_list_screen(img):
        items: List[str] = []
        saw_cancel = False
        for ty in (4, 6, 8, 10):
            name = read_tile_text(img, ty, 6, 18)
            if not name:
                continue
            if name == "CANCEL":
                saw_cancel = True
            else:
                items.append(name)
        arrow_sig = _tile_row_signature(img[11 * TILE_SIZE:12 * TILE_SIZE, 18 * TILE_SIZE:19 * TILE_SIZE])
        has_more = (not saw_cancel) and (bool(items) or arrow_sig in GEN1_SCROLL_ARROW_SIGNATURES)
        return (items, has_more)
    if is_pack_item_list_screen(img) and not is_tm_hm_pack_pocket(img):
        items = []
        saw_cancel = False
        for ty in (2, 4, 6, 8, 10):
            name = read_tile_text(img, ty, 8, 20)
            if not name:
                continue
            if name == "CANCEL" or name.startswith("CANCEL "):
                saw_cancel = True
            else:
                items.append(name)
        has_more = (not saw_cancel) and bool(items)
        return (items, has_more)
    return None


HEALING_ITEM_NAMES = (
    "POTION",
    "SUPER POTION",
    "HYPER POTION",
    "MAX POTION",
    "FULL RESTORE",
    "FRESH WATER",
    "SODA POP",
    "LEMONADE",
    "MOOMOO MILK",
    "GOLD BERRY",
    "ENERGYPOWDER",
    "ENERGY ROOT",
    "BERRY",
    "BERRY JUICE",
)

STATUS_HEAL_ITEM_NAMES = (
    "ANTIDOTE",
    "FULL HEAL",
    "FULL RESTORE",
    "PARLYZ HEAL",
    "AWAKENING",
    "BURN HEAL",
    "ICE HEAL",
    "PSNCUREBERRY",
    "PRZCUREBERRY",
    "MINT BERRY",
    "BURNT BERRY",
    "ICE BERRY",
    "MIRACLEBERRY",
    "HEAL POWDER",
)

HP_RESTORE_AMOUNT_ITEMS: Dict[int, Tuple[str, ...]] = {
    10: ("BERRY",),
    20: ("POTION", "BERRY JUICE"),
    30: ("GOLD BERRY",),
    50: ("SUPER POTION", "FRESH WATER", "ENERGYPOWDER"),
    60: ("SODA POP",),
    80: ("LEMONADE",),
    100: ("MOOMOO MILK",),
    200: ("HYPER POTION", "ENERGY ROOT"),
}


def is_healing_item_name(name: str) -> bool:
    """Return True if an uppercase bag item name is an HP-restoring item."""
    up = re.sub(r"\s*\(X\d+\)\s*$", "", (name or "").strip().upper())
    if not up or up == "CANCEL":
        return False
    if any(b in up for b in ("PSNCURE", "PRZCURE", "MINT BERRY", "BURNT BERRY", "ICE BERRY", "MIRACLEBERRY")):
        return False
    return any(up == target or target in up for target in HEALING_ITEM_NAMES)


def is_status_heal_item_name(name: str) -> bool:
    """Return True if an uppercase bag item name is a status-curing item."""
    up = re.sub(r"\s*\(X\d+\)\s*$", "", (name or "").strip().upper())
    if not up or up == "CANCEL":
        return False
    return any(up == target or target in up for target in STATUS_HEAL_ITEM_NAMES)


STATUS_CURE_MAP: Dict[str, Tuple[str, ...]] = {
    "PSN": ("ANTIDOTE", "PSNCUREBERRY", "FULL HEAL", "FULL RESTORE", "MIRACLEBERRY", "HEAL POWDER"),
    "PAR": ("PARLYZ HEAL", "PRZCUREBERRY", "FULL HEAL", "FULL RESTORE", "MIRACLEBERRY", "HEAL POWDER"),
    "SLP": ("AWAKENING", "MINT BERRY", "FULL HEAL", "FULL RESTORE", "MIRACLEBERRY", "HEAL POWDER"),
    "BRN": ("BURN HEAL", "BURNT BERRY", "FULL HEAL", "FULL RESTORE", "MIRACLEBERRY", "HEAL POWDER"),
    "FRZ": ("ICE HEAL", "ICE BERRY", "FULL HEAL", "FULL RESTORE", "MIRACLEBERRY", "HEAL POWDER"),
}


def is_status_heal_item_for_status(name: str, status: Optional[str] = None) -> bool:
    """Return True if name cures the given status code (or any status if status is None/unknown)."""
    up = re.sub(r"\s*\(X\d+\)\s*$", "", (name or "").strip().upper())
    if not up or up == "CANCEL":
        return False
    st = (status or "").strip().upper()
    if st in STATUS_CURE_MAP:
        return any(up == target or target in up for target in STATUS_CURE_MAP[st])
    return is_status_heal_item_name(up)


PARTY_BOOST_ITEM_NAMES = (
    "RARE CANDY",
    "HP UP",
    "PROTEIN",
    "IRON",
    "CARBOS",
    "CALCIUM",
    "PP UP",
)


def is_party_boost_item_name(name: str) -> bool:
    """Return True if an uppercase bag item name is a level-up or stat-boosting vitamin item used on a party Pokemon."""
    up = re.sub(r"\s*\(X\d+\)\s*$", "", (name or "").strip().upper())
    if not up or up == "CANCEL":
        return False
    return any(up == target or target in up for target in PARTY_BOOST_ITEM_NAMES)


def is_party_consumable_item_name(name: str) -> bool:
    """Return True if an uppercase bag item name is an HP-healing, status-healing, or stat/level-boosting party item."""
    return (
        is_healing_item_name(name)
        or is_status_heal_item_name(name)
        or is_party_boost_item_name(name)
    )


def read_tile_quantity(img: np.ndarray, ty: int, tx_start: int, tx_end: int) -> Optional[int]:
    """Read an integer quantity from 8x8 digit tiles on row ty in [tx_start, tx_end)."""
    y0 = ty * TILE_SIZE
    if img.shape[0] < y0 + TILE_SIZE:
        return None
    max_tx = min(tx_end, img.shape[1] // TILE_SIZE)
    digits: List[str] = []
    for tx in range(tx_start, max_tx):
        tile = img[y0:y0 + TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
        if float((tile >= LIGHT_BACKGROUND).mean()) < 0.45:
            continue
        sig = _tile_row_signature(tile)
        if sig in HUD_DIGIT_TEMPLATES:
            digits.append(str(HUD_DIGIT_TEMPLATES[sig]))
        elif ty == 11 and sig[:6] in HUD_DIGIT_TOP6_TEMPLATES and sig[6] == 0 and sig[7] in (0, 3):
            digits.append(str(HUD_DIGIT_TOP6_TEMPLATES[sig[:6]]))
        elif digits:
            break
    if not digits:
        return None
    val = int("".join(digits))
    return val if 1 <= val <= 999 else None


def read_bag_item_quantities(frame: np.ndarray) -> Optional[List[Tuple[str, Optional[int]]]]:
    """Return [(item_name, quantity_or_None), ...] for visible items on a Gen 1 or Gen 2 bag item list, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if is_gen1_item_list_screen(img):
        out: List[Tuple[str, Optional[int]]] = []
        for ty in (4, 6, 8, 10):
            name = read_tile_text(img, ty, 6, 18)
            if not name or name == "CANCEL":
                continue
            qty = read_tile_quantity(img, ty + 1, 14, 18)
            out.append((name, qty))
        return out
    if is_pack_item_list_screen(img):
        out = []
        for ty in (2, 4, 6, 8, 10):
            name = read_tile_text(img, ty, 8, 20)
            if not name or name == "CANCEL" or name.startswith("CANCEL "):
                continue
            qty = read_tile_quantity(img, ty + 1, 17, 20)
            out.append((name, qty))
        return out
    return None


def read_bag_cursor_row_index(frame: np.ndarray, tracked_index: Optional[int] = None) -> Optional[int]:
    """Return the 0-based visible window row index currently highlighted on a Gen 1 or Gen 2 bag item list, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if is_gen1_item_list_screen(img):
        rows: List[str] = []
        for idx, ty in enumerate((4, 6, 8, 10)):
            name = read_tile_text(img, ty, 6, 18)
            if name:
                rows.append(name)
            if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 5 * TILE_SIZE:6 * TILE_SIZE]):
                return idx
        if tracked_index is not None and 0 <= tracked_index < len(rows):
            return tracked_index
        return None
    if is_pack_item_list_screen(img):
        rows = []
        for idx, ty in enumerate((2, 4, 6, 8, 10)):
            name = read_tile_text(img, ty, 8, 20)
            if not name:
                continue
            row_name = "CANCEL" if (name == "CANCEL" or name.startswith("CANCEL ")) else name
            rows.append(row_name)
            if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 7 * TILE_SIZE:8 * TILE_SIZE]):
                return idx
        cur = read_bag_cursor_row(img, tracked_index=tracked_index)
        if cur in rows:
            return rows.index(cur)
        return None
    return None


def read_bag_cursor_row(frame: np.ndarray, tracked_index: Optional[int] = None) -> Optional[str]:
    """Return the item name (or 'CANCEL') currently highlighted on a Gen 1 or Gen 2 bag item list, or None."""
    if frame is None:
        return None
    img = _grey_frame(frame)
    if is_gen1_item_list_screen(img):
        rows: List[str] = []
        for ty in (4, 6, 8, 10):
            name = read_tile_text(img, ty, 6, 18)
            if name:
                rows.append(name)
            if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 5 * TILE_SIZE:6 * TILE_SIZE]):
                return name if name else None
        if tracked_index is not None and 0 <= tracked_index < len(rows):
            return rows[tracked_index]
        return None
    if is_pack_item_list_screen(img):
        rows: List[str] = []
        for ty in (2, 4, 6, 8, 10):
            name = read_tile_text(img, ty, 8, 20)
            if not name:
                continue
            row_name = "CANCEL" if (name == "CANCEL" or name.startswith("CANCEL ")) else name
            rows.append(row_name)
            if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 7 * TILE_SIZE:8 * TILE_SIZE]):
                return row_name
        if not rows:
            return None
        desc_box = img[13 * TILE_SIZE:17 * TILE_SIZE, TILE_SIZE:14 * TILE_SIZE]
        if int((desc_box < DARK_INK).sum()) <= 10:
            if "CANCEL" in rows:
                return "CANCEL"
            return None
        non_cancel = [r for r in rows if r != "CANCEL"]
        if len(non_cancel) == 1:
            return non_cancel[0]
        line16_hp = read_tile_text(img, 16, 1, 3)
        line16_num = read_tile_quantity(img, 16, 5, 11)
        if line16_hp == "HP" and line16_num is not None:
            candidates = HP_RESTORE_AMOUNT_ITEMS.get(line16_num, ())
            matches = [r for r in non_cancel if any(c == r or r.startswith(c) or c.startswith(r) for c in candidates)]
            if len(matches) == 1:
                return matches[0]
        if tracked_index is not None and 0 <= tracked_index < len(rows) and rows[tracked_index] != "CANCEL":
            return rows[tracked_index]
        if line16_hp != "HP":
            status_matches = [r for r in non_cancel if not is_healing_item_name(r) or is_status_heal_item_name(r)]
            if len(status_matches) == 1:
                return status_matches[0]
        return non_cancel[0] if non_cancel else None
    return None


MULTIPLY_SIGN_SIGNATURE = (0, 0, 68, 40, 16, 40, 68, 0)


def read_quantity_selector_value(frame: np.ndarray) -> Optional[int]:
    """Read the integer quantity (1..99) displayed inside an open quantity selector sub-box, or None."""
    if frame is None or not has_quantity_selector_box(frame):
        return None
    img = _grey_frame(frame)
    max_ty = min(GRID_ROWS, img.shape[0] // TILE_SIZE)
    max_tx = min(GRID_COLS, img.shape[1] // TILE_SIZE)
    if max_ty < GRID_ROWS or max_tx < GRID_COLS:
        return None
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
                    for ty in range(top_ty + 1, bot_ty):
                        for tx in range(left_tx + 1, min(right_tx - 1, max_tx - 2)):
                            sig = _tile_row_signature(tiles[ty][tx])
                            if sig == MULTIPLY_SIGN_SIGNATURE:
                                val = read_tile_quantity(img, ty, tx + 1, min(tx + 4, right_tx))
                                if val is not None and 1 <= val <= 99:
                                    return val
                        val = read_tile_quantity(img, ty, left_tx + 1, min(left_tx + 5, right_tx))
                        if val is not None and 1 <= val <= 99:
                            return val
    return None


def read_menu_cursor_item_name(frame: np.ndarray, tracked_index: Optional[int] = None) -> Optional[str]:
    """Return the uppercase item name currently highlighted by a cursor on a bag list or shop shelf, or None."""
    if frame is None:
        return None
    bag_row = read_bag_cursor_row(frame, tracked_index=tracked_index)
    if bag_row and bag_row != "CANCEL":
        return bag_row
    img = _grey_frame(frame)
    for ty in (2, 4, 6, 8, 10):
        for cursor_tx, text_tx in ((5, 6), (7, 8), (1, 2)):
            if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, cursor_tx * TILE_SIZE:(cursor_tx + 1) * TILE_SIZE]):
                name = read_tile_text(img, ty, text_tx, 19)
                if name and name != "CANCEL" and not name.startswith("CANCEL "):
                    return name
    return None


def is_gen1_item_action_subbox_screen(frame: np.ndarray) -> bool:
    """Return True if a Generation 1 ITEM screen has its USE/TOSS action sub-box open with the cursor on USE."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < 15 * TILE_SIZE or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if is_party_menu_screen(img) or is_pack_screen(img) or has_quantity_selector_box(img):
        return False
    if not all(is_horizontal_border_tile(img[2 * TILE_SIZE:3 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
               for tx in range(5, 19)):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 4 * TILE_SIZE:5 * TILE_SIZE])
               for ty in range(3, 12)):
        return False
    if not all(is_horizontal_border_tile(img[10 * TILE_SIZE:11 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
               for tx in range(14, 19)):
        return False
    if not all(is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 13 * TILE_SIZE:14 * TILE_SIZE])
               for ty in range(11, 14)):
        return False
    return is_cursor_tile(img[11 * TILE_SIZE:12 * TILE_SIZE, 14 * TILE_SIZE:15 * TILE_SIZE])


def is_overworld_pack_use_subbox_screen(frame: np.ndarray) -> bool:
    """Return True if a Generation 2 Pack screen has its action sub-box open with the cursor on the top USE option."""
    if not is_pack_screen(frame) or has_quantity_selector_box(frame):
        return False
    img = _grey_frame(frame)
    if pack_action_box_height(img) < 4:
        return False
    for top_ty in range(3, 10):
        for left_tx in range(10, 16):
            span_end = min(19, left_tx + 5)
            if span_end <= left_tx + 2:
                continue
            if not all(
                is_horizontal_border_tile(img[top_ty * TILE_SIZE:(top_ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
                for tx in range(left_tx + 1, span_end)
            ):
                continue
            if (
                is_cursor_tile(
                    img[(top_ty + 1) * TILE_SIZE:(top_ty + 2) * TILE_SIZE, (left_tx + 1) * TILE_SIZE:(left_tx + 2) * TILE_SIZE]
                )
                and read_tile_text(img, top_ty + 1, left_tx + 2, 19) == "USE"
            ):
                return True
    return False


def is_item_use_subbox_screen(frame: np.ndarray) -> bool:
    """Return True if a Gen 1 or Gen 2 item action sub-box (USE/TOSS, USE/QUIT, or USE/GIVE/TOSS/QUIT) is open with cursor on USE."""
    return (
        is_overworld_pack_use_subbox_screen(frame)
        or is_gen1_item_action_subbox_screen(frame)
    )


def has_party_subbox(frame: np.ndarray) -> bool:
    """Return True if an interactive sub-box (YES/NO, 4-move deletion box, or party action menu) is drawn on the party screen."""
    if not is_party_menu_screen(frame):
        return False
    img = _grey_frame(frame)
    right_v = sum(
        1 for ty in range(3, 12)
        if is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 19 * TILE_SIZE:20 * TILE_SIZE])
    )
    if right_v < 3:
        return False
    return any(
        all(
            is_horizontal_border_tile(img[top_ty * TILE_SIZE:(top_ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE])
            for tx in range(15, 19)
        )
        for top_ty in range(2, 9)
    )


def read_party_move_forget_list(frame: np.ndarray) -> Optional[Tuple[List[str], Optional[str]]]:
    """Return (move_names, highlighted_move) when the 4-move forget box ('Which move should be forgotten?') is open on a party screen, or None."""
    if not is_party_menu_screen(frame):
        return None
    img = _grey_frame(frame)
    if (
        all(is_horizontal_border_tile(img[2 * TILE_SIZE:3 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]) for tx in range(5, 19))
        and sum(1 for ty in range(3, 11) if is_vertical_border_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 19 * TILE_SIZE:20 * TILE_SIZE])) >= 6
    ):
        moves: List[str] = []
        cur_move: Optional[str] = None
        for ty in (4, 6, 8, 10):
            name = read_tile_text(img, ty, 7, 19)
            if name and name != "-":
                moves.append(name)
                if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 6 * TILE_SIZE:7 * TILE_SIZE]):
                    cur_move = name
        if len(moves) >= 2:
            return (moves, cur_move)
    if (
        all(is_horizontal_border_tile(img[7 * TILE_SIZE:8 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]) for tx in range(5, 19))
        and all(is_horizontal_border_tile(img[12 * TILE_SIZE:13 * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]) for tx in range(5, 19))
    ):
        moves = []
        cur_move = None
        for ty in (8, 9, 10, 11):
            name = read_tile_text(img, ty, 6, 19)
            if name and name != "-":
                moves.append(name)
                if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 5 * TILE_SIZE:6 * TILE_SIZE]):
                    cur_move = name
        if len(moves) >= 2:
            return (moves, cur_move)
    return None


def read_party_menu_members(frame: np.ndarray) -> List[Tuple[int, str, Optional[Tuple[int, int]]]]:
    """Return [(slot_idx, name, hp_tuple_or_None), ...] for visible party slots on a Gen 1 or Gen 2 party menu screen."""
    if not is_party_menu_screen(frame):
        return []
    img = _grey_frame(frame)
    members: List[Tuple[int, str, Optional[Tuple[int, int]]]] = []
    gen1_name0 = read_tile_text(img, 0, 3, 14)
    if gen1_name0 and gen1_name0 != "CANCEL":
        for idx, name_ty in enumerate((0, 2, 4, 6, 8, 10)):
            name = read_tile_text(img, name_ty, 3, 14)
            if not name or name == "CANCEL" or name.startswith("CANCEL "):
                break
            hp = _read_tile_hp_fraction(img, name_ty + 1, 16)
            members.append((idx, name, hp))
        return members
    for idx, row_ty in enumerate((1, 3, 5, 7, 9, 11)):
        if read_tile_text(img, row_ty, 1, 8) == "CANCEL":
            break
        name = read_tile_text(img, row_ty, 3, 13)
        if not name or name == "CANCEL" or name.startswith("CANCEL "):
            break
        hp = _read_tile_hp_fraction(img, row_ty, 16)
        members.append((idx, name, hp))
    return members


def read_party_menu_cursor_slot(frame: np.ndarray) -> Optional[int]:
    """Return the 0-based party slot index currently highlighted by the solid cursor on a party screen, or None."""
    if not is_party_menu_screen(frame):
        return None
    img = _grey_frame(frame)
    for idx, ty in enumerate((1, 3, 5, 7, 9, 11)):
        if is_cursor_tile(img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, 0:TILE_SIZE]):
            return idx
    return None


def read_party_menu_compatibility(frame: np.ndarray) -> List[Tuple[int, str, Optional[str]]]:
    """Return [(slot_idx, name, compat_or_None), ...] where compat_or_None is 'ABLE', 'NOT ABLE', or 'LEARNED' when visible on a TM/HM/stone party screen."""
    if not is_party_menu_screen(frame):
        return []
    img = _grey_frame(frame)
    out: List[Tuple[int, str, Optional[str]]] = []
    gen1_name0 = read_tile_text(img, 0, 3, 14)
    if gen1_name0 and gen1_name0 != "CANCEL":
        for idx, name_ty in enumerate((0, 2, 4, 6, 8, 10)):
            name = read_tile_text(img, name_ty, 3, 14)
            if not name or name == "CANCEL" or name.startswith("CANCEL "):
                break
            compat_row = read_tile_text(img, name_ty + 1, 10, 20)
            compat: Optional[str] = None
            if "NOT ABLE" in compat_row or "UNABLE" in compat_row:
                compat = "NOT ABLE"
            elif "ABLE" in compat_row:
                compat = "ABLE"
            elif "LEARNED" in compat_row:
                compat = "LEARNED"
            out.append((idx, name, compat))
        return out
    for idx, row_ty in enumerate((1, 3, 5, 7, 9, 11)):
        if read_tile_text(img, row_ty, 1, 8).startswith("CANC"):
            break
        name = read_tile_text(img, row_ty, 3, 13)
        if not name or name == "CANCEL" or name.startswith("CANCEL "):
            break
        compat_row = read_tile_text(img, row_ty + 1, 10, 20) if row_ty + 1 < 13 else ""
        compat = None
        if "NOT ABLE" in compat_row or "UNABLE" in compat_row:
            compat = "NOT ABLE"
        elif "ABLE" in compat_row:
            compat = "ABLE"
        elif "LEARNED" in compat_row:
            compat = "LEARNED"
        out.append((idx, name, compat))
    return out


def is_overworld_confirmation_subbox(frame: np.ndarray) -> bool:
    """Return True if a 2-option YES/NO or HEAL/CANCEL confirmation sub-box is open over an overworld bottom dialogue box with the cursor on the top affirmative option."""
    if frame is None:
        return False
    img = _grey_frame(frame)
    if img.shape[0] < DIALOGUE_BOX_BOTTOM or img.shape[1] < GRID_COLS * TILE_SIZE:
        return False
    if not has_bottom_dialogue_box(img):
        return False
    if (
        is_party_menu_screen(img)
        or is_pack_screen(img)
        or is_start_menu_screen(img)
        or is_gen1_item_list_screen(img)
    ):
        return False
    for cursor_ty in (7, 8, 9):
        for cursor_tx in range(11, 17):
            if not is_cursor_tile(
                img[cursor_ty * TILE_SIZE:(cursor_ty + 1) * TILE_SIZE,
                    cursor_tx * TILE_SIZE:(cursor_tx + 1) * TILE_SIZE]
            ):
                continue
            top_opt = read_tile_text(img, cursor_ty, cursor_tx + 1, 20)
            bot_opt = (
                read_tile_text(img, cursor_ty + 2, cursor_tx + 1, 20)
                if cursor_ty + 2 < 12
                else ""
            )
            if (top_opt == "YES" and bot_opt == "NO") or (top_opt == "HEAL" and bot_opt == "CANCEL"):
                return True
    return False


def read_move_learning_dialogue_status(frame: np.ndarray) -> Optional[Tuple[str, str, str]]:
    """Return (status, pokemon_name, move_name) from a move-learning dialogue box ('learned', 'knows', 'forgot', 'did_not_learn'), or None."""
    if frame is None or not has_bottom_dialogue_box(frame):
        return None
    l1, l2 = read_dialogue_box_lines(frame, tx_start=1, tx_end=19)
    if not l1 or not l2:
        return None
    m_learned = re.match(r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+learn(?:ed)?$", l1)
    if m_learned and re.match(r"^[A-Z][A-Z0-9 -]{2,14}$", l2):
        return ("learned", m_learned.group(1).strip(), l2.strip())
    m_knows = re.match(r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+knows$", l1)
    if m_knows and re.match(r"^[A-Z][A-Z0-9 -]{2,14}$", l2):
        return ("knows", m_knows.group(1).strip(), l2.strip())
    m_already = re.match(r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+already$", l1)
    if m_already:
        m_k2 = re.match(r"^knows\s+([A-Z][A-Z0-9 -]{2,14})$", l2)
        if m_k2:
            return ("knows", m_already.group(1).strip(), m_k2.group(1).strip())
    m_forgot = re.match(r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+forgo(?:t)?$", l1)
    if m_forgot and re.match(r"^[A-Z][A-Z0-9 -]{2,14}$", l2):
        return ("forgot", m_forgot.group(1).strip(), l2.strip())
    m_did_not = re.match(r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+did\s+not$", l1)
    if m_did_not:
        m_dn2 = re.match(r"^learn\s+([A-Z][A-Z0-9 -]{2,14})$", l2)
        if m_dn2:
            return ("did_not_learn", m_did_not.group(1).strip(), m_dn2.group(1).strip())
    return None
