import os
import math
import time
import json
import cv2
import numpy as np

# =====================================================================
# 1. DROIDCAM STREAM CONFIGURATION
# =====================================================================
PHONE_IP = "172.30.194.152"   # Change to the IP shown on your phone
DROIDCAM_PORT = "4747"
DROIDCAM_STREAM_URL = f"http://{PHONE_IP}:{DROIDCAM_PORT}/video"
CAMERA_SOURCE = DROIDCAM_STREAM_URL

# =====================================================================
# 2. CANONICAL GEOMETRY & MEASURED SPECIFICATIONS
# =====================================================================
CANONICAL_SIZE = 512
BOARD_CENTER = (256.0, 256.0)
JSON_LOG_PATH = "robostrike_session_log.json"

CANONICAL_MARKER_CENTROIDS = {
    0: np.array([55.0, 55.0], dtype=np.float32),    # Top-Left
    1: np.array([457.0, 55.0], dtype=np.float32),   # Top-Right
    2: np.array([457.0, 457.0], dtype=np.float32),  # Bottom-Right
    3: np.array([55.0, 457.0], dtype=np.float32),   # Bottom-Left
}

# Physical scale: outer scoring ring at measured 7.0 equals 200.0 px
MAX_BOARD_RADIUS_MEASURED = 7.0
CANONICAL_OUTER_RADIUS_PX = 200.0
SCALE_FACTOR = MAX_BOARD_RADIUS_MEASURED / CANONICAL_OUTER_RADIUS_PX

# Progressive Hit Multiplier Settings
ENABLE_STREAK_MULTIPLIER = True
STREAK_INCREMENT = 0.10     # +10% per consecutive on-target dart
MAX_MULTIPLIER = 1.50       # Capped at 1.5x

# Dart detection spatial filters
DART_MIN_DIST_PX = 38.0
DART_CONFIRM_FRAMES = 8
MAX_DARTS = 15

# Measured Anchors: (radius_limit, score_value)
# Ordered from center bullseye outwards
ANCHORS = [
    (2.1, 80),
    (3.0, 60),
    (3.8, 50),
    (4.5, 40),
    (5.3, 30),
    (6.1, 20),
    (7.0, 10),
]


def compute_progressive_score(x: float, y: float, dart_seq_num: int) -> tuple[int, int, float, float, float]:
    """
    Computes score using piecewise linear plateau-decay between measured boundaries,
    and applies progressive streak multipliers.
    """
    r_px = math.hypot(x - BOARD_CENTER[0], y - BOARD_CENTER[1])
    r_meas = r_px * SCALE_FACTOR

    # Out of bounds / Miss
    if r_meas > MAX_BOARD_RADIUS_MEASURED or r_meas < 0.0:
        return 0, 0, 1.0, r_px, r_meas

    # Bullseye Center Plateau
    if r_meas <= ANCHORS[0][0]:
        base_score = ANCHORS[0][1]
    else:
        # Linear decay across measured interval
        base_score = 0
        for i in range(len(ANCHORS) - 1):
            r_inner, s_inner = ANCHORS[i]
            r_outer, s_outer = ANCHORS[i + 1]

            if r_inner < r_meas <= r_outer:
                decay_fraction = (r_meas - r_inner) / (r_outer - r_inner)
                interpolated = s_inner - decay_fraction * (s_inner - s_outer)
                base_score = int(math.ceil(interpolated))
                break

    # Calculate progressive multiplier
    if ENABLE_STREAK_MULTIPLIER and base_score > 0:
        multiplier = min(1.0 + (dart_seq_num - 1) * STREAK_INCREMENT, MAX_MULTIPLIER)
    else:
        multiplier = 1.0

    final_score = int(round(base_score * multiplier))
    return final_score, base_score, multiplier, r_px, r_meas


def save_session_log(locked_darts: list, total_score: int):
    session_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_score": total_score,
        "total_darts_hit": len(locked_darts),
        "progressive_streak_enabled": ENABLE_STREAK_MULTIPLIER,
        "darts": [
            {
                "dart_index": idx,
                "canonical_x": round(d["x"], 2),
                "canonical_y": round(d["y"], 2),
                "radial_distance_px": round(d["r_px"], 2),
                "measured_distance": round(d["r_meas"], 2),
                "base_score": d["base_score"],
                "multiplier": round(d["multiplier"], 2),
                "final_score": d["final_score"]
            }
            for idx, d in enumerate(locked_darts, start=1)
        ]
    }
    try:
        with open(JSON_LOG_PATH, "w") as f:
            json.dump(session_data, f, indent=4)
    except Exception as e:
        print(f"[WARN] Failed to write JSON log: {e}")


def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.008
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return lambda gray: detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = 0.008
        return lambda gray: cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)


# =====================================================================
# DUAL-STAGE RED DOMINANCE FILTER
# =====================================================================
def extract_red_dart_mask(warped_bgr):
    b, g, r = cv2.split(warped_bgr)

    # 1. Chromatic Dominance Filter (Red strongly exceeds Green and Blue)
    diff_rg = cv2.subtract(r, g)
    diff_rb = cv2.subtract(r, b)
    rgb_red = (diff_rg > 42) & (diff_rb > 42) & (r > 90)
    rgb_mask = rgb_red.astype(np.uint8) * 255

    # 2. Strict Red HSV Filter
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    mask_low = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([7, 255, 255]))
    mask_high = cv2.inRange(hsv, np.array([172, 110, 70]), np.array([180, 255, 255]))
    hsv_mask = cv2.bitwise_or(mask_low, mask_high)

    combined_red = cv2.bitwise_and(rgb_mask, hsv_mask)

    # Scoring circular boundary mask (r = 202px)
    roi = np.zeros_like(combined_red)
    cv2.circle(roi, (256, 256), 202, 255, -1)
    return cv2.bitwise_and(combined_red, roi)


# =====================================================================
# 3. MAIN RUNTIME ENGINE
# =====================================================================
def main():
    detect_aruco = create_aruco_detector()

    locked_darts = []
    candidate_darts = []
    total_score = 0
    show_mask = False
    baseline_mask = None

    save_session_log(locked_darts, total_score)
    print(f"[INFO] Initialized empty session log at {JSON_LOG_PATH}")
    print(f"[INFO] Connecting to DroidCam video stream: {CAMERA_SOURCE} ...")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream at {CAMERA_SOURCE}.")
        return

    print("[SUCCESS] DroidCam connected. Progressive Scoring Engine active.")

    last_valid_H = None
    last_marker_positions = {}
    last_warped_board = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_aruco(gray)
            detected_centroids = {}

            if ids is not None and len(ids) > 0:
                for marker_corners, raw_id in zip(corners, np.asarray(ids).flatten()):
                    m_id = int(raw_id)
                    if m_id in CANONICAL_MARKER_CENTROIDS:
                        pts = np.asarray(marker_corners, dtype=np.float32).reshape((4, 2))
                        detected_centroids[m_id] = np.mean(pts, axis=0).astype(np.float32)

            num_markers = len(detected_centroids)
            H = None

            # 4-marker perspective homography
            if num_markers == 4:
                src_pts = np.array([detected_centroids[i] for i in range(4)], dtype=np.float32)
                dst_pts = np.array([CANONICAL_MARKER_CENTROIDS[i] for i in range(4)], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    last_valid_H = H.copy()
                    last_marker_positions = {k: v.copy() for k, v in detected_centroids.items()}
            # 3-marker affine fallback
            elif num_markers == 3:
                valid_ids = list(detected_centroids.keys())[:3]
                src_pts = np.array([detected_centroids[i] for i in valid_ids], dtype=np.float32)
                dst_pts = np.array([CANONICAL_MARKER_CENTROIDS[i] for i in valid_ids], dtype=np.float32)
                M_aff = cv2.getAffineTransform(src_pts, dst_pts)
                H = np.vstack([M_aff, [0.0, 0.0, 1.0]])
                last_valid_H = H.copy()
                for k, v in detected_centroids.items():
                    last_marker_positions[k] = v.copy()
            # 1 or 2-marker translation fallback
            elif num_markers in (1, 2) and last_valid_H is not None:
                common_ids = [k for k in detected_centroids if k in last_marker_positions]
                if common_ids:
                    deltas = [detected_centroids[k] - last_marker_positions[k] for k in common_ids]
                    mean_dx, mean_dy = np.mean(deltas, axis=0)
                    T_shift = np.array([
                        [1.0, 0.0, -float(mean_dx)],
                        [0.0, 1.0, -float(mean_dy)],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)
                    H = last_valid_H @ T_shift
                    last_valid_H = H.copy()
                    for k in common_ids:
                        last_marker_positions[k] = detected_centroids[k].copy()
                else:
                    H = last_valid_H.copy()

            if H is not None:
                warped = cv2.warpPerspective(frame, H, (CANONICAL_SIZE, CANONICAL_SIZE))

                if num_markers >= 3 and len(locked_darts) < MAX_DARTS:
                    raw_red_mask = extract_red_dart_mask(warped)

                    # Initial clean baseline calibration
                    if baseline_mask is None and len(locked_darts) == 0:
                        baseline_mask = cv2.dilate(raw_red_mask, morph_kernel, iterations=2)
                        print("[INFO] Clean baseline captured.")

                    if baseline_mask is not None:
                        active_mask = cv2.subtract(raw_red_mask, baseline_mask)
                    else:
                        active_mask = raw_red_mask

                    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_OPEN, morph_kernel)
                    active_mask = cv2.dilate(active_mask, morph_kernel, iterations=1)

                    contours, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    current_centroids = []

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 30.0 or area > 1200.0:
                            continue

                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            current_centroids.append((float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])))

                    filtered = []
                    for (cx, cy) in current_centroids:
                        if not any(math.hypot(cx - fx, cy - fy) < DART_MIN_DIST_PX for (fx, fy) in filtered):
                            filtered.append((cx, cy))

                    for (cx, cy) in filtered:
                        if any(math.hypot(cx - d["x"], cy - d["y"]) < DART_MIN_DIST_PX for d in locked_darts):
                            continue

                        matched = False
                        for cand in candidate_darts:
                            if math.hypot(cx - cand["x"], cy - cand["y"]) < DART_MIN_DIST_PX:
                                cand["x"] = 0.7 * cand["x"] + 0.3 * cx
                                cand["y"] = 0.7 * cand["y"] + 0.3 * cy
                                cand["count"] += 1
                                cand["last_seen"] = time.time()
                                matched = True

                                if cand["count"] >= DART_CONFIRM_FRAMES and len(locked_darts) < MAX_DARTS:
                                    dart_seq_num = len(locked_darts) + 1
                                    final_s, base_s, mult, r_px, r_meas = compute_progressive_score(
                                        cand["x"], cand["y"], dart_seq_num
                                    )

                                    locked_darts.append({
                                        "x": cand["x"],
                                        "y": cand["y"],
                                        "final_score": final_s,
                                        "base_score": base_s,
                                        "multiplier": mult,
                                        "r_px": r_px,
                                        "r_meas": r_meas
                                    })
                                    total_score += final_s
                                    print(f"🎯 [PROGRESSIVE HIT #{dart_seq_num}] "
                                          f"Dist={r_meas:.2f} | Base={base_s} | "
                                          f"x{mult:.2f} -> +{final_s} pts | Total: {total_score}")
                                    save_session_log(locked_darts, total_score)
                                break

                        if not matched:
                            candidate_darts.append({"x": cx, "y": cy, "count": 1, "last_seen": time.time()})

                    now = time.time()
                    candidate_darts = [c for c in candidate_darts if c["count"] < DART_CONFIRM_FRAMES and (now - c["last_seen"]) < 1.2]

                    if show_mask:
                        warped = cv2.cvtColor(active_mask, cv2.COLOR_GRAY2BGR)

                last_warped_board = warped.copy()
            else:
                warped = last_warped_board.copy() if last_valid_H is not None else cv2.resize(frame, (CANONICAL_SIZE, CANONICAL_SIZE))

            # Overlay confirmed darts on canonical warped view
            for i, d in enumerate(locked_darts, start=1):
                pt = (int(round(d["x"])), int(round(d["y"])))
                cv2.circle(warped, pt, 8, (255, 255, 0), 2)
                cv2.circle(warped, pt, 3, (0, 0, 255), -1)
                cv2.putText(warped, f"#{i}:{d['final_score']}pts", (pt[0] + 10, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

            # Build Telemetry HUD Panel
            hud = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), (22, 22, 28), dtype=np.uint8)
            cv2.putText(hud, "TECHNOZION ROBOSTRIKE", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.line(hud, (25, 56), (487, 56), (80, 80, 100), 1)

            status_str = f"LOCKED ({num_markers}/4)" if num_markers >= 3 else f"TRACKING ({num_markers}/4)"
            cv2.putText(hud, f"STATUS: {status_str}", (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 2, cv2.LINE_AA)

            calib_str = "CALIBRATED" if baseline_mask is not None else "READY (PRESS R)"
            cv2.putText(hud, f"BASELINE: {calib_str}", (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 220), 1, cv2.LINE_AA)

            cv2.putText(hud, f"DARTS ON TARGET: {len(locked_darts)}", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(hud, f"TOTAL SCORE: {total_score}", (25, 205), cv2.FONT_HERSHEY_DUPLEX, 0.95, (50, 255, 50), 2, cv2.LINE_AA)

            # Progressive Telemetry Breakdown
            if locked_darts:
                last_d = locked_darts[-1]
                hit_str = f"+{last_d['final_score']} (dist={last_d['r_meas']:.2f})"
                sub_str = f"Base: {last_d['base_score']} | Streak: x{last_d['multiplier']:.2f}"
            else:
                hit_str = "WAITING FOR THROW..."
                sub_str = "Progressive Decay & Streak Active"

            cv2.putText(hud, "LAST HIT:", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(hud, hit_str, (25, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(hud, sub_str, (25, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 220, 255), 1, cv2.LINE_AA)

            cv2.putText(hud, "[R] Reset/Calibrate   [M] Toggle Mask   [Q] Quit", (25, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 160), 1, cv2.LINE_AA)

            combined = np.hstack((warped, hud))
            cv2.imshow("Technozion Robostrike - Vision Scoring Engine", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("r"):
                locked_darts.clear()
                candidate_darts.clear()
                total_score = 0
                baseline_mask = None
                save_session_log(locked_darts, total_score)
                print("🔄 [RESET] Darts cleared. Baseline reference recalibrated.")
            elif key == ord("m"):
                show_mask = not show_mask

    except KeyboardInterrupt:
        print("\n[INFO] Saving session log...")

    finally:
        save_session_log(locked_darts, total_score)
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()