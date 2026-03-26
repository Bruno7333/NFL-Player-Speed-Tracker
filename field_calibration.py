import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')   # off-screen rendering — no GUI conflict with OpenCV
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Constants — Phase 1 classification (theta space)
# ─────────────────────────────────────────────
# In HoughLines: theta ∈ [0, π)
#   theta ≈ π/2  → horizontal line  → sideline
#   theta ≈ 0/π  → vertical line    → yard line
SIDELINE_ANGLE_THRESHOLD  = np.deg2rad(20)  # within 20° of ±π/2 → sideline (after normalisation)
YARD_LINE_ANGLE_THRESHOLD = np.deg2rad(75)  # within 75° of 0    → yard line (after normalisation)

# ─────────────────────────────────────────────
# Constants — mini-map rendering
# ─────────────────────────────────────────────
MINIMAP_W      = 200
MINIMAP_H      = 106
MINIMAP_ORIGIN = (10, 10)

# ─────────────────────────────────────────────
# Constants — field geometry
# ─────────────────────────────────────────────
FIELD_WIDTH    = 53.3    # yards — distance between the two sidelines
VP_DEPTH_YARDS = 76.35   # How far away the camera is from the sideline
                         # Close Sideline is roughly 50 yds
                         # Far Sideline is roughly 103.3 yds
                         # Currently chose 76.35 for roughly halfway down the field
                         # See if we can find a way to detect distance from sideline from video feed.
                         # Dont know how to do this yet, try to fix later. Estimate is good enough for now.


# ═════════════════════════════════════════════
# Utility — colour helpers
# ═════════════════════════════════════════════

def tab20_bgr(group_idx, total_groups):
    """Return an OpenCV BGR tuple matching the matplotlib tab20 colour for group_idx."""
    cmap  = matplotlib.colormaps.get_cmap('tab20').resampled(max(total_groups, 1))
    r, g, b, _ = cmap(group_idx)
    return (int(b * 255), int(g * 255), int(r * 255))


# ═════════════════════════════════════════════
# Utility — rho/theta normalisation
# ═════════════════════════════════════════════

def normalize_rho_theta(rho, theta):
    """
    Fold theta into (-π/2, π/2] so parallel lines always land on the same
    branch of the sinusoid.  Used for grouping and heatmap display only;
    the original (rho, theta) is kept for drawing.
    """
    if theta > np.pi / 2:
        theta -= np.pi
        rho    = -rho
    return rho, theta


# ═════════════════════════════════════════════
# Utility — rho/theta → slope-intercept
# ═════════════════════════════════════════════

def rho_theta_to_mb(rho, theta):
    """
    Convert a HoughLines (rho, theta) line to slope-intercept (m, b).

    The normal-form equation is:  x·cos(θ) + y·sin(θ) = ρ
    Rearranging for y:            y = (ρ - x·cos(θ)) / sin(θ)
    So: m = -cos(θ)/sin(θ),  b = ρ/sin(θ)

    Returns (m, b) or None for lines where sin(θ) ≈ 0 (near-vertical,
    which HoughLines expresses as θ ≈ 0 or π).  Near-vertical lines are
    skipped because the grouping logic requires a finite slope.
    """
    sin_t = np.sin(theta)
    if abs(sin_t) < 1e-6:
        return None   # nearly vertical — undefined in slope-intercept form
    m = -np.cos(theta) / sin_t
    b = rho / sin_t
    return m, b


def rho_theta_to_endpoints(rho, theta, frame_width, frame_height):
    """
    Compute two pixel endpoints for a (rho, theta) line clipped to the frame.
    Uses x = 0 and x = frame_width as the two anchor x-values, then solves for y.
    Falls back to horizontal extent via y = 0 / y = frame_height for near-vertical lines.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    if abs(sin_t) > 1e-6:
        # Solve y at x = 0 and x = frame_width
        x1, y1 = 0,           int(rho / sin_t)
        x2, y2 = frame_width, int((rho - frame_width * cos_t) / sin_t)
    else:
        # Near-vertical: solve x at y = 0 and y = frame_height
        x1, y1 = int(rho / cos_t), 0
        x2, y2 = int(rho / cos_t), frame_height

    return x1, y1, x2, y2


# ═════════════════════════════════════════════
# 1.1  Line classification
# ═════════════════════════════════════════════

def classify_line(theta):
    # After normalisation theta ∈ (-π/2, π/2]
    # Sideline: near ±π/2 (horizontal in image)
    if abs(abs(theta) - np.pi / 2) < SIDELINE_ANGLE_THRESHOLD:
        return 'sideline'
    # Yard line: near 0 (vertical in image)
    if abs(theta) < YARD_LINE_ANGLE_THRESHOLD:
        return 'yard_line'
    return 'unknown'


def label_yard_lines(yard_lines, sideline_x_left=None, sideline_x_right=None):
    _ = sideline_x_left, sideline_x_right
    labeled = []
    for i, (m, b) in enumerate(yard_lines):
        labeled.append((m, b, i * 5))
    return labeled


# ═════════════════════════════════════════════
# 1.2  Homography
# ═════════════════════════════════════════════

def compute_homography(src_points, dst_points):
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    H, _ = cv.findHomography(src, dst)
    return H


# ═════════════════════════════════════════════
# 1.3  Field coordinate transformer
# ═════════════════════════════════════════════

def pixel_to_field(px, py, H):
    pt     = np.array([[[px, py]]], dtype=np.float32)
    result = cv.perspectiveTransform(pt, H)
    return float(result[0][0][0]), float(result[0][0][1])


# ═════════════════════════════════════════════
# 1.4  Overlay helpers
# ═════════════════════════════════════════════

def draw_calibration_overlay(frame, sidelines, labeled_yard_lines, H_valid):
    """
    sidelines          : list of (rho, theta, color_bgr)
    labeled_yard_lines : list of (rho, theta, yard_num, color_bgr)
    Colors match the tab20 palette used in the heatmap so groups are
    visually consistent across both windows.
    """
    h, w = frame.shape[:2]

    for rho, theta, color in sidelines:
        x1, y1, x2, y2 = rho_theta_to_endpoints(rho, theta, w, h)
        cv.line(frame, (x1, y1), (x2, y2), color, 4)

    for rho, theta, yard_num, color in labeled_yard_lines:
        x1, y1, x2, y2 = rho_theta_to_endpoints(rho, theta, w, h)
        cv.line(frame, (x1, y1), (x2, y2), color, 2)
        # Place label where the line crosses the vertical midpoint of the frame
        mid_y = h // 2
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        if abs(cos_t) > 1e-6:
            label_x = int((rho - mid_y * sin_t) / cos_t)
        else:
            label_x = int(rho)   # near-vertical fallback
        label_y = mid_y - 10
        cv.putText(frame, str(yard_num), (label_x, label_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status_text  = "H valid" if H_valid else "H stale"
    status_color = (0, 200, 0) if H_valid else (0, 0, 200)
    cv.putText(frame, status_text, (10, h - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


def draw_minimap(frame, yard_lines_field_x, field_length_visible=100):
    x0, y0 = MINIMAP_ORIGIN

    cv.rectangle(frame,
                 (x0, y0),
                 (x0 + MINIMAP_W, y0 + MINIMAP_H),
                 (34, 139, 34), -1)

    cv.rectangle(frame,
                 (x0, y0),
                 (x0 + MINIMAP_W, y0 + MINIMAP_H),
                 (255, 255, 255), 1)

    for fx in yard_lines_field_x:
        if 0 <= fx <= field_length_visible:
            px = x0 + int((fx / field_length_visible) * MINIMAP_W)
            cv.line(frame, (px, y0), (px, y0 + MINIMAP_H), (255, 255, 255), 1)


# ═════════════════════════════════════════════
# Helper — rho/theta line intersection
# ═════════════════════════════════════════════

def intersect_rho_theta(rho1, theta1, rho2, theta2):
    """
    Intersect two lines given in HoughLines normal form.
    Solves: [cos θ₁  sin θ₁] [x]   [ρ₁]
            [cos θ₂  sin θ₂] [y] = [ρ₂]
    Returns (x, y) or None if lines are parallel.
    """
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]], dtype=np.float64)
    b = np.array([rho1, rho2], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    xy = np.linalg.solve(A, b)
    return float(xy[0]), float(xy[1])


# ═════════════════════════════════════════════
# Helper — vanishing point from yard lines
# ═════════════════════════════════════════════

def compute_vanishing_point(yard_line_groups):
    """
    Intersect every pair of yard line groups to find where they converge.
    Returns the median intersection as (vx, vy), or None if < 2 groups.
    """
    pts = []
    for i in range(len(yard_line_groups)):
        for j in range(i + 1, len(yard_line_groups)):
            ga, gb = yard_line_groups[i], yard_line_groups[j]
            pt = intersect_rho_theta(ga['rho'], ga['theta'], gb['rho'], gb['theta'])
            if pt is not None:
                pts.append(pt)
    if not pts:
        return None
    pts = np.array(pts, dtype=np.float64)
    return (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))


# ═════════════════════════════════════════════
# Helper — build anchor points (hybrid VP method)
# ═════════════════════════════════════════════

def build_anchor_points_hybrid(primary_sl, labeled_yl_display, vp):
    """
    Build src/dst correspondences using one detected sideline + vanishing point.

    For each labeled yard line:
      P_i = primary_sideline ∩ yard_line_i         → field (yard_num, 0)
      Q_i = P_i + t*(VP - P_i)                     → field (yard_num, FIELD_WIDTH)
            where t = FIELD_WIDTH / (FIELD_WIDTH + VP_DEPTH_YARDS)

    Returns (src_pts, dst_pts, rho_second) or (None, None, None).
    rho_second is the synthesised second sideline's rho (for display).
    """
    if primary_sl is None or vp is None or len(labeled_yl_display) < 2:
        return None, None, None

    rho_sl, theta_sl = primary_sl['rho'], primary_sl['theta']
    vp_arr = np.array(vp, dtype=np.float64)
    t = FIELD_WIDTH / (FIELD_WIDTH + VP_DEPTH_YARDS)

    src_pts, dst_pts, q_rhos = [], [], []

    for rho_yl, theta_yl, yard_num, _ in labeled_yl_display:
        P = intersect_rho_theta(rho_sl, theta_sl, rho_yl, theta_yl)
        if P is None:
            continue
        P_arr = np.array(P, dtype=np.float64)
        Q_arr = P_arr + t * (vp_arr - P_arr)

        src_pts.append(P_arr)
        dst_pts.append([float(yard_num), 0.0])
        src_pts.append(Q_arr)
        dst_pts.append([float(yard_num), FIELD_WIDTH])
        q_rhos.append(Q_arr[0] * np.cos(theta_sl) + Q_arr[1] * np.sin(theta_sl))

    if len(src_pts) < 4:
        return None, None, None

    rho_second = float(np.median(q_rhos))
    return (np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            rho_second)


# ═════════════════════════════════════════════
# Rho/Theta heatmap debug window
# ═════════════════════════════════════════════

def show_rho_theta_heatmap(raw_lines, groups=None, yl_fit=None, yl_fit2=None):
    """
    Show a 2D density heatmap of all (theta, rho) pairs from HoughLines.
    If `groups` is provided (post Phase-1.1), each cluster is overlaid
    with a distinct colour and labelled by group index + class.
    yl_fit  — (a, b) first-pass fit through all yard-line candidates (dashed white)
    yl_fit2 — (a, b) refined fit after outlier removal (solid yellow)
    """
    if raw_lines is None or len(raw_lines) == 0:
        return

    # Normalise all raw lines to (-π/2, π/2] for display
    norm_pairs = [normalize_rho_theta(r, t) for r, t in raw_lines[:, 0, :]]
    rhos_all   = np.array([p[0] for p in norm_pairs])
    thetas_all = np.array([p[1] for p in norm_pairs])

    rho_min = rhos_all.min() - 100
    rho_max = rhos_all.max() + 100

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Background density heatmap — theta -90°→90°, rho padded ±100
    ax.hist2d(
        np.degrees(thetas_all), rhos_all,
        bins=[180, 300],
        range=[[-90, 90], [rho_min, rho_max]],
        cmap='inferno'
    )

    ax.set_xlim(-90, 90)
    ax.set_ylim(rho_min, rho_max)
    ax.set_xlabel('theta (degrees, normalised)')
    ax.set_ylabel('rho (pixels)')
    ax.set_title(f'HoughLines accumulator — {len(raw_lines)} raw lines, '
                 f'{len(groups) if groups else 0} groups')

    if groups:
        # One colour per group, cycling through a qualitative palette
        palette = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(groups), 1))

        # Class → marker shape so class is readable even when colours overlap
        marker_for = {'sideline': 's', 'yard_line': '^', 'unknown': 'o'}
        edge_for   = {'sideline': 'deepskyblue',
                      'yard_line': 'limegreen',
                      'unknown':  'lightgray'}

        for idx, g in enumerate(groups):
            rts = g['rho_theta']
            t_deg = np.degrees([rt[1] for rt in rts])
            r_arr = [rt[0] for rt in rts]
            label_str = g.get('label', 'unknown')

            ax.scatter(
                t_deg, r_arr,
                s=40,
                color=palette(idx),
                edgecolors=edge_for.get(label_str, 'white'),
                linewidths=0.8,
                marker=marker_for.get(label_str, 'o'),
                alpha=0.85,
                label=f'G{idx} {label_str} ({len(rts)})',
                zorder=3
            )

        # Draw yard-line fit line across the full theta range
        t_vals = np.linspace(-90, 90, 200)
        if yl_fit is not None:
            a, b = yl_fit
            r_vals = a * np.deg2rad(t_vals) + b
            ax.plot(t_vals, r_vals, color='white', linewidth=1.5,
                    linestyle='--', label='fit 1 (all candidates)', zorder=4)
        if yl_fit2 is not None:
            a2, b2 = yl_fit2
            r_vals2 = a2 * np.deg2rad(t_vals) + b2
            ax.plot(t_vals, r_vals2, color='yellow', linewidth=2.0,
                    linestyle='-', label='fit 2 (inliers only)', zorder=5)

        # Compact legend — show at most 20 entries before truncating
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 20:
            handles, labels = handles[:20], labels[:20]
            labels[-1] = '… (truncated)'
        ax.legend(handles, labels, loc='upper right',
                  fontsize=7, ncol=2, framealpha=0.6)
    else:
        # No group info — plain cyan dots
        ax.scatter(np.degrees(thetas_all), rhos_all,
                   s=6, c='cyan', alpha=0.4, linewidths=0, label='raw lines')
        ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    heatmap_bgr = cv.cvtColor(buf, cv.COLOR_RGBA2BGR)
    cv.imshow('Rho/Theta Heatmap', heatmap_bgr)
