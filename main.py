import cv2 as cv
import numpy as np
from collections import deque
from ultralytics import YOLO
from player_tracking import foot_point
from field_calibration import (
    normalize_rho_theta,
    rho_theta_to_mb,
    classify_line,
    label_yard_lines,
    compute_homography,
    pixel_to_field,
    draw_calibration_overlay,
    draw_minimap,
    compute_vanishing_point,
    build_anchor_points_hybrid,
    show_rho_theta_heatmap,
    tab20_bgr,
    YARD_LINE_ANGLE_THRESHOLD,
    FIELD_WIDTH,
    MINIMAP_ORIGIN,
    MINIMAP_W,
    MINIMAP_H,
)

# ─────────────────────────────────────────────
# Constants — white mask
# ─────────────────────────────────────────────
lower_white = np.array([50, 0, 140])
upper_white = np.array([180, 80, 255])
#WHITE_PROXIMITY_PX = 1  # keep pixels within this many pixels of white
                        # Used to dilate right before input into canny filter

# ─────────────────────────────────────────────
# Constants — green (field) mask
# ─────────────────────────────────────────────
lower_green = np.array([30, 30, 30])
upper_green = np.array([95, 255, 255])
GREEN_PROXIMITY_PX = 10   # keep white pixels within this many pixels of green

# ─────────────────────────────────────────────
# Constants — Canny edge detection
# ─────────────────────────────────────────────
CANNY_THRESH1 = 125   # lower hysteresis threshold
CANNY_THRESH2 = 175   # upper hysteresis threshold

# ─────────────────────────────────────────────
# Constants — Hough Lines
# ─────────────────────────────────────────────
HOUGH_THRESHOLD = 125  # minimum votes for a line to be accepted

# ─────────────────────────────────────────────
# Constants — line grouping (rho/theta space)
# ─────────────────────────────────────────────
ANGLE_THRESHOLD = np.deg2rad(10)   # max angular difference between lines in a group (radians)
RHO_THRESHOLD   = 50               # max rho difference between lines in a group (pixels)

# ─────────────────────────────────────────────
# Constants — video source
# ─────────────────────────────────────────────
VIDEO_PATH = 'Images/TyreekHill.mkv'   # set to the name and location of your desired video input

# ─────────────────────────────────────────────
# Constants — heatmap throttle
# ─────────────────────────────────────────────
HEATMAP_UPDATE_INTERVAL = 1    # redraw the heatmap debug window every N frames
CROP_BOTTOM = 0              # pixels to mask from the bottom before detection (scoreboard)

# ─────────────────────────────────────────────
# Mode switch
# ─────────────────────────────────────────────
# True  → testing mode:     tab20 colours, heatmap, Canny edges, white mask windows all shown
# False → production mode:  yard lines = red, sidelines = blue; debug windows hidden
TESTING_MODE = False

# ─────────────────────────────────────────────
# Constants — Phase 2 player detection / tracking
# ─────────────────────────────────────────────
YOLO_MODEL      = 'yolov8m.pt'
YOLO_CONF       = 0.35
YOLO_IOU        = 0.45
MAX_LOST_FRAMES = 30

# ─────────────────────────────────────────────
# Constants — yard-line fit / temporal filter
# ─────────────────────────────────────────────
YL_FIT_THRESHOLD = 150   # max vertical rho residual (pixels) to be an inlier

# ─────────────────────────────────────────────
# Constants — Phase 3 speed calculation
# ─────────────────────────────────────────────
SPEED_WINDOW_FRAMES = 5
SPEED_HISTORY_LEN   = 10
SPEED_SMOOTH_LEN    = 5
SPEED_MAX_MPH       = 25.0
SPEED_MIN_YARDS     = 0.1
YDS_PER_S_TO_MPH    = 2.04545


# ═════════════════════════════════════════════
# Live video pipeline
# ═════════════════════════════════════════════

_kgreen = GREEN_PROXIMITY_PX * 2 + 1
GREEN_DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_kgreen, _kgreen))

#_kwhite = WHITE_PROXIMITY_PX * 2 + 1
#WHITE_DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_kwhite, _kwhite))

cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video source: {VIDEO_PATH}")

frame_index = 0
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
paused = True   # start paused; press Space to play
prev_avg_yl_theta = None   # rolling average theta for yard lines (radians, normalised)
YL_TEMPORAL_THRESHOLD  = np.deg2rad(30)   # max deviation from prev-frame average
SL_RHO_THRESHOLD       = 40              # max frame-to-frame sideline rho shift (px)
SL_THETA_THRESHOLD     = np.deg2rad(8)  # max frame-to-frame sideline angle shift
prev_sl_rho    = None   # sideline rho from previous frame
prev_sl_theta  = None   # sideline theta from previous frame
prev_H         = None   # last successfully computed homography
H_stale_frames = 0      # consecutive frames since H was last computed fresh
H_MAX_STALE    = 5      # carry prev_H for at most this many frames

yolo          = YOLO(YOLO_MODEL)
selected_id   = None
lost_counter  = 0
click_pending = None

video_fps         = cap.get(cv.CAP_PROP_FPS) or 29.97
pos_history       = deque(maxlen=SPEED_HISTORY_LEN)
speed_history     = deque(maxlen=SPEED_SMOOTH_LEN)
current_speed_mph = 0.0
prev_selected_id  = None

def on_mouse(event, x, y, flags, param):
    global click_pending
    if event == cv.EVENT_LBUTTONDOWN:
        click_pending = (x, y)

cv.namedWindow('NFL Live (HoughLines)')
cv.setMouseCallback('NFL Live (HoughLines)', on_mouse)

while cap.isOpened():
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        break   # end of file or read error
    frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LINEAR)

    frame_height, frame_width = frame.shape[:2]

    # ── White-mask → Canny edge map ──────────────────────────────────────
    hsv_frame  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    green_mask = cv.inRange(hsv_frame, lower_green, upper_green)
    green_prox = cv.dilate(green_mask, GREEN_DILATE_KERNEL)
    white_mask = cv.bitwise_and(cv.inRange(hsv_frame, lower_white, upper_white),
                                green_prox)
    white_gray = cv.bitwise_and(
        cv.cvtColor(frame, cv.COLOR_BGR2GRAY), white_mask
    )
    # Blank out the scoreboard rectangle so it doesn't feed into Canny/Hough
    if CROP_BOTTOM > 0:
        white_gray[frame_height - CROP_BOTTOM:, :] = 0
    #white_dilated = cv.dilate(white_gray, WHITE_DILATE_KERNEL)
    blurred = cv.GaussianBlur(white_gray, (5, 5), 0)
    edges   = cv.Canny(blurred, CANNY_THRESH1, CANNY_THRESH2)

    # ── HoughLines detection ──────────────────────────────────────────────
    # Returns lines as (rho, theta) in an array of shape (N, 1, 2)
    raw_lines = cv.HoughLines(
        edges,
        1,            # rho resolution (pixels)
        np.pi / 180,  # theta resolution (radians)
        HOUGH_THRESHOLD
    )

    # ── Fit a line through raw yard-line candidates (pre-grouping) ───────
    yl_fit  = None
    yl_fit2 = None
    if raw_lines is not None:
        yl_candidates = [
            normalize_rho_theta(line[0][0], line[0][1]) for line in raw_lines
            if abs(normalize_rho_theta(line[0][0], line[0][1])[1]) < YARD_LINE_ANGLE_THRESHOLD
        ]
        if len(yl_candidates) >= 2:
            cand_thetas = np.array([rt[1] for rt in yl_candidates])
            cand_rhos   = np.array([rt[0] for rt in yl_candidates])
            if cand_thetas.std() > 1e-6:   # skip degenerate (all-same theta) input
                a, b = np.polyfit(cand_thetas, cand_rhos, 1)
                yl_fit = (a, b)

                # Remove outliers and refit — threshold is vertical rho residual
                inliers = np.abs((a * cand_thetas + b) - cand_rhos) < YL_FIT_THRESHOLD
                if inliers.sum() >= 2 and cand_thetas[inliers].std() > 1e-6:
                    a2, b2 = np.polyfit(cand_thetas[inliers], cand_rhos[inliers], 1)
                    yl_fit2 = (a2, b2)
                else:
                    yl_fit2 = yl_fit

    # ── Temporal yard-line angle filter ──────────────────────────────────
    if raw_lines is not None and prev_avg_yl_theta is not None:
        def _keep(line):
            _, theta_n = normalize_rho_theta(line[0][0], line[0][1])
            if abs(theta_n) < YARD_LINE_ANGLE_THRESHOLD:
                return abs(theta_n - prev_avg_yl_theta) < YL_TEMPORAL_THRESHOLD
            return True
        raw_lines = np.array([l for l in raw_lines if _keep(l)])
        if len(raw_lines) == 0:
            raw_lines = None

    # ── Convert (rho, theta) → (m, b) and group co-linear lines ──────────
    groups = []

    if raw_lines is not None:
        for line in raw_lines:
            rho_raw, theta_raw = line[0]
            # Normalise to (-π/2, π/2] for grouping and display consistency
            rho, theta = normalize_rho_theta(rho_raw, theta_raw)
            result = rho_theta_to_mb(rho_raw, theta_raw)
            if result is None:
                continue  # skip near-vertical lines with undefined slope
            m, b = result

            found = False
            for group in groups:
                if (abs(theta - group['theta']) < ANGLE_THRESHOLD and
                        abs(rho - group['rho']) < RHO_THRESHOLD):
                    group['rho_theta'].append((rho, theta))
                    n = len(group['rho_theta'])
                    group['rho']   = (group['rho']   * (n - 1) + rho)   / n
                    group['theta'] = (group['theta'] * (n - 1) + theta) / n
                    # Recompute m, b from updated average rho/theta for classification
                    result = rho_theta_to_mb(group['rho'], group['theta'])
                    if result:
                        group['m'], group['b'] = result
                    found = True
                    break
            if not found:
                groups.append({'rho': rho, 'theta': theta, 'm': m, 'b': b,
                               'rho_theta': [(rho, theta)]})

    # ── Phase 1.1 — Classify grouped lines ──────────────────────────────
    sideline_groups  = []
    yard_line_groups = []

    for group in groups:
        label = classify_line(group['theta'])
        group['label'] = label
        if label == 'sideline':
            sideline_groups.append(group)
        elif label == 'yard_line':
            yard_line_groups.append(group)

    # Assign each group the same tab20 colour used in the heatmap
    for idx, g in enumerate(groups):
        g['color_bgr'] = tab20_bgr(idx, len(groups))

    # ── Filter yard line groups against the pre-grouping fit ─────────────
    if yl_fit2 is not None and yard_line_groups:
        a, b = yl_fit2
        yard_line_groups = [
            g for g in yard_line_groups
            if abs((a * g['theta'] + b) - g['rho']) < YL_FIT_THRESHOLD
        ]

    # ── Heatmap — only redrawn every HEATMAP_UPDATE_INTERVAL frames ──────
    if TESTING_MODE and frame_index % HEATMAP_UPDATE_INTERVAL == 0:
        show_rho_theta_heatmap(raw_lines, groups, yl_fit=yl_fit, yl_fit2=yl_fit2)

    def group_x_at_mid(g):
        m, b  = g['m'], g['b']
        mid_y = frame_height / 2.0
        if abs(m) < 1e-6:
            return 0.0
        return (mid_y - b) / m

    sideline_x_left  = 0
    sideline_x_right = frame_width
    if len(sideline_groups) >= 2:
        sl_sorted        = sorted(sideline_groups, key=group_x_at_mid)
        sideline_x_left  = int(group_x_at_mid(sl_sorted[0]))
        sideline_x_right = int(group_x_at_mid(sl_sorted[-1]))

    yl_sorted_groups = sorted(yard_line_groups, key=group_x_at_mid)
    yl_mb_list       = [(g['m'], g['b']) for g in yl_sorted_groups]
    labeled_yl       = label_yard_lines(yl_mb_list, sideline_x_left, sideline_x_right)

    labeled_yl_display = []
    for (m, b, yard_num), g in zip(labeled_yl, yl_sorted_groups):
        rho, theta = g['rho_theta'][0]
        labeled_yl_display.append((rho, theta, yard_num, g['color_bgr']))

    # ── Phase 1.2 — Homography (hybrid: one sideline + VP) ───────────────
    H = None
    primary_sl  = max(sideline_groups, key=lambda g: len(g['rho_theta'])) \
                  if sideline_groups else None

    # ── Sideline temporal filter — reject implausible frame-to-frame jumps ─
    if primary_sl is not None and prev_sl_rho is not None:
        if (abs(primary_sl['rho']   - prev_sl_rho)   > SL_RHO_THRESHOLD or
                abs(primary_sl['theta'] - prev_sl_theta) > SL_THETA_THRESHOLD):
            primary_sl = None   # reject; H will fall back to prev_H
    if primary_sl is not None:
        prev_sl_rho   = primary_sl['rho']
        prev_sl_theta = primary_sl['theta']

    vp          = compute_vanishing_point(yard_line_groups)
    rho_second  = None

    if primary_sl is not None:
        src_pts, dst_pts, rho_second = build_anchor_points_hybrid(
            primary_sl, labeled_yl_display, vp
        )
        if src_pts is not None:
            H = compute_homography(src_pts, dst_pts)

    # ── Homography persistence — carry prev frame for up to H_MAX_STALE ──
    if H is not None:
        prev_H         = H
        H_stale_frames = 0
    elif prev_H is not None and H_stale_frames < H_MAX_STALE:
        H              = prev_H
        H_stale_frames += 1
    else:
        H_stale_frames += 1

    # ── Build display lists ──────────────────────────────────────────────
    sidelines_display = []
    if primary_sl is not None:
        sidelines_display.append((primary_sl['rho'], primary_sl['theta'],
                                   primary_sl['color_bgr']))
    if rho_second is not None and primary_sl is not None:
        sidelines_display.append((rho_second, primary_sl['theta'], (0, 255, 255)))

    # ── Phase 1.4a — Draw calibration overlay ────────────────────────────
    if TESTING_MODE:
        sl_draw = sidelines_display
        yl_draw = labeled_yl_display
    else:
        sl_draw = [(rho, theta, (255, 0, 0)) for rho, theta, _ in sidelines_display]
        yl_draw = [(rho, theta, yard_num, (0, 0, 255))
                   for rho, theta, yard_num, _ in labeled_yl_display]
    draw_calibration_overlay(frame, sl_draw, yl_draw, H_stale_frames == 0)

    # ── Phase 1.4b — Draw bird's-eye mini-map ────────────────────────────
    if H is not None:
        yard_lines_field_x = []
        for g in yl_sorted_groups:
            px_mid = group_x_at_mid(g)
            py_mid = frame_height / 2.0
            fx = pixel_to_field(px_mid, py_mid, H)[0] + 5
            yard_lines_field_x.append(fx)
        n_yl = max(len(yard_lines_field_x), 1)
        field_length_visible = (n_yl - 1) * 5 + 10
        draw_minimap(frame, yard_lines_field_x, field_length_visible)
    else:
        draw_minimap(frame, [])

    # ── Phase 2 — YOLO + ByteTrack ───────────────────────────────────────
    results = yolo.track(
        frame,
        persist   = True,
        tracker   = 'bytetrack.yaml',
        classes   = [0],
        conf      = YOLO_CONF,
        iou       = YOLO_IOU,
        verbose   = False
    )

    boxes = []
    if results[0].boxes is not None and results[0].boxes.id is not None:
        xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids  = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = list(zip(xyxy, ids))

    # ── Consume pending click → select nearest player ────────────────────
    if click_pending is not None and boxes:
        cx, cy        = click_pending
        click_pending = None
        best_id, best_dist = None, float('inf')

        for (x1, y1, x2, y2), tid in boxes:
            fx, fy = foot_point(x1, y1, x2, y2)
            dist   = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_id = dist, int(tid)

        selected_id  = best_id
        lost_counter = 0

        if selected_id != prev_selected_id:
            pos_history.clear()
            speed_history.clear()
            current_speed_mph = 0.0
        prev_selected_id = selected_id

    # ── ID loss watchdog ─────────────────────────────────────────────────
    current_ids = {int(tid) for _, tid in boxes}

    if selected_id is not None:
        if selected_id in current_ids:
            lost_counter = 0
        else:
            lost_counter += 1
            if lost_counter > MAX_LOST_FRAMES:
                selected_id  = None
                lost_counter = 0

    # ── Map selected player foot → field coords ───────────────────────────
    player_field_pos = None
    player_foot_py   = None   # raw pixel Y of selected player's foot
    player_foot_px   = None   # raw pixel X of selected player's foot

    if selected_id is not None and selected_id in current_ids and H is not None:
        for (x1, y1, x2, y2), tid in boxes:
            if int(tid) == selected_id:
                fx_px, fy_px = foot_point(x1, y1, x2, y2)
                player_field_pos = pixel_to_field(fx_px, fy_px, H)
                player_foot_py   = fy_px
                player_foot_px   = fx_px
                break

    # ── Phase 3 — position history + speed calculation ───────────────────
    if player_field_pos is not None and selected_id is not None \
            and selected_id in current_ids:
        pos_history.append(player_field_pos)

    if len(pos_history) >= SPEED_WINDOW_FRAMES:
        x0, y0 = pos_history[-SPEED_WINDOW_FRAMES]
        x1, y1 = pos_history[-1]
        distance_yards = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        time_seconds   = SPEED_WINDOW_FRAMES / video_fps
        if distance_yards < SPEED_MIN_YARDS:
            speed_mph = 0.0
        else:
            speed_mph = (distance_yards / time_seconds) * YDS_PER_S_TO_MPH
        if speed_mph <= SPEED_MAX_MPH:
            speed_history.append(speed_mph)
        if speed_history:
            current_speed_mph = float(np.median(speed_history))

    # ── Draw all players (grey foot dots) ────────────────────────────────
    for (x1, y1, x2, y2), tid in boxes:
        if int(tid) != selected_id:
            fx, fy = foot_point(x1, y1, x2, y2)
            cv.circle(frame, (fx, fy), 4, (180, 180, 180), -1)

    # ── Draw selected player ─────────────────────────────────────────────
    if selected_id is not None and selected_id in current_ids:
        for (x1, y1, x2, y2), tid in boxes:
            if int(tid) == selected_id:
                fx, fy = foot_point(x1, y1, x2, y2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.circle(frame, (fx, fy), 5, (0, 255, 0), -1)
                if player_field_pos is not None:
                    label = f"ID {selected_id}  ({player_field_pos[0]:.1f}yd, {player_field_pos[1]:.1f}yd)"
                else:
                    label = f"ID {selected_id}  (no H)"
                cv.putText(frame, f"{current_speed_mph:.1f} mph", (x1, y1 - 28),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(frame, label, (x1, y1 - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                break

    elif selected_id is not None:
        cv.putText(frame, f"ID {selected_id} -- Player lost ({lost_counter}/{MAX_LOST_FRAMES})",
                   (10, frame_height - 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    if selected_id is None:
        cv.putText(frame, "Click a player to track",
                   (10, frame_height - 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ── Minimap: blue line for detected sideline at centre ───────────────
    sl_minimap_y = MINIMAP_ORIGIN[1] + MINIMAP_H // 2
    cv.line(frame,
            (MINIMAP_ORIGIN[0], sl_minimap_y),
            (MINIMAP_ORIGIN[0] + MINIMAP_W, sl_minimap_y),
            (255, 0, 0), 1)

    # ── Phase 2 — player dot on minimap ──────────────────────────────────
    if (player_field_pos is not None and H is not None
            and player_foot_py is not None and player_foot_px is not None
            and primary_sl is not None):
        # X: field coordinate from homography
        fp_x = player_field_pos[0]
        fp_x_clamped = max(0.0, min(fp_x, field_length_visible))
        dot_px = MINIMAP_ORIGIN[0] + int((fp_x_clamped / field_length_visible) * MINIMAP_W)

        # Y: pixel offset from detected sideline, centred on minimap midline
        sin_sl = np.sin(primary_sl['theta'])
        cos_sl = np.cos(primary_sl['theta'])
        if abs(sin_sl) > 1e-6:
            sl_y_at_foot = (primary_sl['rho'] - player_foot_px * cos_sl) / sin_sl
        else:
            sl_y_at_foot = primary_sl['rho'] / cos_sl
        offset_px = player_foot_py - sl_y_at_foot   # + = below sideline in frame
        dot_py = sl_minimap_y + int(offset_px * MINIMAP_H / frame_height)
        dot_py = max(MINIMAP_ORIGIN[1], min(dot_py, MINIMAP_ORIGIN[1] + MINIMAP_H))

        cv.circle(frame, (dot_px, dot_py), 5, (0, 255, 0), -1)
        cv.circle(frame, (dot_px, dot_py), 5, (255, 255, 255), 1)

    # ── Display ───────────────────────────────────────────────────────────
    cv.imshow('NFL Live (HoughLines)', frame)
    if TESTING_MODE:
        cv.imshow('NFL Edges', edges)
        cv.imshow('NFL B&W', white_gray)

    # Update rolling yard-line theta for next frame
    if yard_line_groups:
        prev_avg_yl_theta = float(np.mean([g['theta'] for g in yard_line_groups]))

    # Key handling — Space toggles play/pause; frame-step only when paused
    if paused:
        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord('d'):
                cap.release()
                cv.destroyAllWindows()
                exit()
            elif key == ord(' '):
                paused = False
                frame_index += 1
                break
            elif key == ord('e'):
                frame_index = min(frame_index + 1, total_frames - 1)
                break
            elif key == ord('q'):
                frame_index = max(frame_index - 1, 0)
                if hasattr(yolo, 'predictor') and yolo.predictor is not None:
                    for tracker in getattr(yolo.predictor, 'trackers', []):
                        tracker.reset()
                break
            elif key == ord('c'):
                selected_id  = None
                lost_counter = 0
                pos_history.clear()
                speed_history.clear()
                current_speed_mph = 0.0
    else:
        key = cv.waitKey(1) & 0xFF
        if key == ord('d'):
            cap.release()
            cv.destroyAllWindows()
            exit()
        elif key == ord(' '):
            paused = True
        elif key == ord('c'):
            selected_id  = None
            lost_counter = 0
            pos_history.clear()
            speed_history.clear()
            current_speed_mph = 0.0
        else:
            frame_index += 1
        if frame_index >= total_frames:
            frame_index = total_frames - 1
            paused = True


cap.release()
cv.destroyAllWindows()
