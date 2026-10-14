import os
import re
import math
import time
import json
import ctypes
from ctypes import (
    c_int, c_uint, c_ulong, c_char_p, c_void_p, POINTER, Structure, byref
)
import subprocess
import cv2
import numpy as np

# =====================================================================
# 1. MATHEMATICAL & GEOMETRIC SPECIFICATIONS & JSON LOGGING
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

RING_RADII = [10, 40, 50, 80, 90, 120, 130, 160, 170, 190, 200]

LOWER_GREEN = np.array([32, 60, 60])
UPPER_GREEN = np.array([90, 255, 255])

DART_MIN_DIST_PX = 22.0
DART_CONFIRM_FRAMES = 5
MAX_DARTS = 15


def compute_dart_score(x: float, y: float) -> tuple[int, float]:
    r = math.hypot(x - BOARD_CENTER[0], y - BOARD_CENTER[1])
    if r >= 200.0 or r < 0.0:
        return 0, r

    k = int(math.floor(r / 40.0))
    delta = r % 40.0
    B_k = 100 - 20 * k

    if delta <= 10.0:
        score = int(B_k)
    else:
        score = int(math.ceil(B_k - (20.0 / 30.0) * (delta - 10.0)))

    return score, r


def save_session_log(locked_darts: list, total_score: int):
    """Saves or updates the session JSON file with current cart positions and scores."""
    session_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_score": total_score,
        "total_darts_hit": len(locked_darts),
        "darts": [
            {
                "dart_index": idx,
                "canonical_x": round(d["x"], 2),
                "canonical_y": round(d["y"], 2),
                "radial_distance_px": round(d["r"], 2),
                "score": d["score"]
            }
            for idx, d in enumerate(locked_darts, start=1)
        ]
    }
    try:
        with open(JSON_LOG_PATH, "w") as f:
            json.dump(session_data, f, indent=4)
    except Exception as e:
        print(f"[WARN] Failed to write JSON log: {e}")


# =====================================================================
# 2. DIRECT LIBX11 WINDOW-ID CAPTURE
# =====================================================================
class XWindowAttributes(Structure):
    _fields_ = [
        ("x", c_int), ("y", c_int),
        ("width", c_int), ("height", c_int),
        ("border_width", c_int), ("depth", c_int),
        ("visual", c_void_p), ("root", c_ulong),
        ("class_", c_int), ("bit_gravity", c_int),
        ("win_gravity", c_int), ("backing_store", c_int),
        ("backing_planes", c_ulong), ("backing_pixel", c_ulong),
        ("save_under", c_int), ("colormap", c_ulong),
        ("map_installed", c_int), ("map_state", c_int),
        ("all_event_masks", c_ulong), ("your_event_mask", c_ulong),
        ("do_not_propagate_mask", c_ulong), ("override_redirect", c_int),
        ("screen", c_void_p),
    ]


class XImage(Structure):
    _fields_ = [
        ("width", c_int), ("height", c_int),
        ("xoffset", c_int), ("format", c_int),
        ("data", c_void_p), ("byte_order", c_int),
        ("bitmap_unit", c_int), ("bitmap_bit_order", c_int),
        ("bitmap_pad", c_int), ("depth", c_int),
        ("bytes_per_line", c_int), ("bits_per_pixel", c_int),
        ("red_mask", c_ulong), ("green_mask", c_ulong), ("blue_mask", c_ulong),
        ("obdata", c_void_p),
    ]


XErrorHandler = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p)


class XWaylandWindowCapturer:
    ZPIXMAP = 2
    ALL_PLANES = 0xFFFFFFFFFFFFFFFF

    def __init__(self):
        self.x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
        self.x11.XOpenDisplay.argtypes = [c_char_p]
        self.x11.XOpenDisplay.restype = c_void_p
        self.x11.XGetWindowAttributes.argtypes = [c_void_p, c_ulong, POINTER(XWindowAttributes)]
        self.x11.XGetWindowAttributes.restype = c_int
        self.x11.XGetImage.argtypes = [c_void_p, c_ulong, c_int, c_int, c_uint, c_uint, c_ulong, c_int]
        self.x11.XGetImage.restype = POINTER(XImage)
        self.x11.XDestroyImage.argtypes = [POINTER(XImage)]
        self.x11.XDestroyImage.restype = c_int

        self._err_handler = XErrorHandler(lambda dpy, err: 0)
        self.x11.XSetErrorHandler(self._err_handler)
        self.display = self.x11.XOpenDisplay(None)
        self.win_id = None

    def find_gazebo_window_id(self) -> int | None:
        try:
            for pattern in ["Gazebo", "dartboard", "gz sim", "Ignition"]:
                res = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", pattern],
                    capture_output=True, text=True, timeout=0.6
                )
                win_ids = [w.strip() for w in res.stdout.splitlines() if w.strip()]
                best_wid, best_area = None, 0
                for wid_str in win_ids:
                    wid = int(wid_str)
                    attrs = XWindowAttributes()
                    if self.display and self.x11.XGetWindowAttributes(self.display, wid, byref(attrs)):
                        area = attrs.width * attrs.height
                        if attrs.width >= 300 and attrs.height >= 300 and area > best_area:
                            best_area = area
                            best_wid = wid
                if best_wid is not None:
                    self.win_id = best_wid
                    return best_wid
        except Exception:
            pass
        return None

    def grab_frame(self) -> np.ndarray | None:
        if not self.display:
            return None
        if self.win_id is None:
            self.find_gazebo_window_id()
            if self.win_id is None:
                return None

        attrs = XWindowAttributes()
        if not self.x11.XGetWindowAttributes(self.display, self.win_id, byref(attrs)):
            self.win_id = None
            return None

        w, h = attrs.width, attrs.height
        if w < 100 or h < 100:
            return None

        ximg_ptr = self.x11.XGetImage(self.display, self.win_id, 0, 0, w, h, self.ALL_PLANES, self.ZPIXMAP)
        if not ximg_ptr:
            self.win_id = None
            return None

        try:
            ximg = ximg_ptr.contents
            bpl = ximg.bytes_per_line
            total_bytes = bpl * h
            buf = (ctypes.c_ubyte * total_bytes).from_address(ximg.data)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, bpl // 4, 4))
            return arr[:, :w, :3].copy()
        finally:
            self.x11.XDestroyImage(ximg_ptr)


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


def main():
    capturer = XWaylandWindowCapturer()
    detect_aruco = create_aruco_detector()

    locked_darts = []
    candidate_darts = []
    total_score = 0
    last_lookup = 0.0
    show_mask = False

    # Instantly create/initialize the JSON log file on startup
    save_session_log(locked_darts, total_score)
    print(f"[INFO] Initialized empty session log at {JSON_LOG_PATH}")

    last_valid_H = None
    last_marker_positions = {}
    last_warped_board = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)

    print("[INFO] Starting Technozion Robostrike Vision Engine...")

    try:
        while True:
            if capturer.win_id is None and (time.time() - last_lookup > 1.5):
                capturer.find_gazebo_window_id()
                last_lookup = time.time()

            frame = capturer.grab_frame()
            if frame is None:
                frame = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
                is_black = True
            else:
                is_black = (np.max(frame) == 0)

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

            if num_markers == 4:
                src_pts = np.array([detected_centroids[i] for i in range(4)], dtype=np.float32)
                dst_pts = np.array([CANONICAL_MARKER_CENTROIDS[i] for i in range(4)], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    last_valid_H = H.copy()
                    last_marker_positions = {k: v.copy() for k, v in detected_centroids.items()}
            elif num_markers == 3:
                valid_ids = list(detected_centroids.keys())[:3]
                src_pts = np.array([detected_centroids[i] for i in valid_ids], dtype=np.float32)
                dst_pts = np.array([CANONICAL_MARKER_CENTROIDS[i] for i in valid_ids], dtype=np.float32)
                M_aff = cv2.getAffineTransform(src_pts, dst_pts)
                H = np.vstack([M_aff, [0.0, 0.0, 1.0]])
                last_valid_H = H.copy()
                for k, v in detected_centroids.items():
                    last_marker_positions[k] = v.copy()
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
                    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
                    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

                    roi_mask = np.zeros_like(mask)
                    cv2.circle(roi_mask, (256, 256), 205, 255, -1)
                    mask = cv2.bitwise_and(mask, roi_mask)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    current_centroids = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if 1.5 <= area < 1200.0:
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
                                cand["x"] = 0.6 * cand["x"] + 0.4 * cx
                                cand["y"] = 0.6 * cand["y"] + 0.4 * cy
                                cand["count"] += 1
                                cand["last_seen"] = time.time()
                                matched = True

                                if cand["count"] >= DART_CONFIRM_FRAMES and len(locked_darts) < MAX_DARTS:
                                    score, r_dist = compute_dart_score(cand["x"], cand["y"])
                                    locked_darts.append({
                                        "x": cand["x"], "y": cand["y"], "score": score, "r": r_dist
                                    })
                                    total_score += score
                                    print(f"[HIT #{len(locked_darts)}] Pos=({cand['x']:.1f},{cand['y']:.1f}) | +{score} pts")
                                    save_session_log(locked_darts, total_score)
                                break

                        if not matched:
                            candidate_darts.append({"x": cx, "y": cy, "count": 1, "last_seen": time.time()})

                    now = time.time()
                    candidate_darts = [c for c in candidate_darts if c["count"] < DART_CONFIRM_FRAMES and (now - c["last_seen"]) < 1.5]

                    if show_mask:
                        warped = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                last_warped_board = warped.copy()
            else:
                warped = last_warped_board.copy() if last_valid_H is not None else cv2.resize(frame, (CANONICAL_SIZE, CANONICAL_SIZE))

            for i, d in enumerate(locked_darts, start=1):
                pt = (int(round(d["x"])), int(round(d["y"])))
                cv2.circle(warped, pt, 7, (0, 255, 0), 2)
                cv2.circle(warped, pt, 2, (0, 0, 255), -1)
                cv2.putText(warped, f"#{i}:{d['score']}", (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # HUD rendering
            hud = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), (22, 22, 28), dtype=np.uint8)
            cv2.putText(hud, "TECHNOZION ROBOSTRIKE", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.line(hud, (25, 56), (487, 56), (80, 80, 100), 1)

            status_str = f"LOCKED ({num_markers}/4) [JSON ACTIVE]" if num_markers >= 3 else f"TRACKING ({num_markers}/4)"
            cv2.putText(hud, f"STATUS: {status_str}", (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(hud, f"DARTS ON TARGET: {len(locked_darts)}", (25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(hud, f"TOTAL SCORE: {total_score}", (25, 178), cv2.FONT_HERSHEY_DUPLEX, 0.95, (50, 255, 50), 2, cv2.LINE_AA)

            cv2.putText(hud, "[R] Reset   [M] Toggle Mask   [Q] Quit", (25, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 160), 1, cv2.LINE_AA)

            combined = np.hstack((warped, hud))
            cv2.imshow("Technozion Robostrike - Vision Scoring Engine", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("r"):
                locked_darts.clear()
                candidate_darts.clear()
                total_score = 0
                save_session_log(locked_darts, total_score)
            elif key == ord("m"):
                show_mask = not show_mask

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted via Ctrl+C. Saving session log...")

    finally:
        # Guarantee final log save on exit (Ctrl+C, 'q', or window close)
        # If no darts were detected, locked_darts will be empty [], writing an empty darts array with score 0.
        save_session_log(locked_darts, total_score)
        print(f"[INFO] Session saved to {JSON_LOG_PATH} (Darts recorded: {len(locked_darts)}, Total Score: {total_score})")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()