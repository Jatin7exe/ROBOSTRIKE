import cv2
import numpy as np

CANONICAL_SIZE = 512
CENTER = (256, 256)
RING_RADII = [10, 40, 50, 80, 90, 120, 130, 160, 170, 190, 200]

MARKER_CENTERS = {
    0: (55, 55),    # Top-Left
    1: (457, 55),   # Top-Right
    2: (457, 457),  # Bottom-Right
    3: (55, 457),   # Bottom-Left
}


def generate_aruco_tile(aruco_dict, marker_id: int, size_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    else:
        return cv2.aruco.drawMarker(aruco_dict, marker_id, size_px)


def main():
    # Initialize dark charcoal canvas
    board = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), (35, 35, 35), dtype=np.uint8)

    # Outer Red Border: (30, 30) to (482, 482)
    cv2.rectangle(board, (30, 30), (482, 482), (0, 0, 255), 3)

    # Draw Concentric Scoring Rings (outermost to innermost)
    for idx, r in enumerate(sorted(RING_RADII, reverse=True)):
        if r in (40, 80, 120, 160, 200):
            fill_color = (245, 245, 245) if (r // 40) % 2 == 1 else (55, 55, 55)
            cv2.circle(board, CENTER, r, fill_color, -1, cv2.LINE_AA)
        cv2.circle(board, CENTER, r, (0, 180, 255), 1, cv2.LINE_AA)

    # Bullseye (r <= 10)
    cv2.circle(board, CENTER, 10, (0, 0, 220), -1, cv2.LINE_AA)
    cv2.circle(board, CENTER, 10, (255, 255, 255), 1, cv2.LINE_AA)

    # Place 4 ArUco Markers (DICT_4X4_50, IDs 0-3) inside high-contrast white backing boxes
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 36   # 36x36 px marker
    box_half = 24      # 48x48 px white backing box -> 6px white quiet zone on all sides

    for m_id, (cx, cy) in MARKER_CENTERS.items():
        # White backing box
        cv2.rectangle(
            board,
            (cx - box_half, cy - box_half),
            (cx + box_half, cy + box_half),
            (255, 255, 255),
            -1
        )
        # ArUco marker centered at (cx, cy)
        tile_gray = generate_aruco_tile(aruco_dict, m_id, marker_size)
        tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
        half_m = marker_size // 2
        board[cy - half_m : cy + half_m, cx - half_m : cx + half_m] = tile_bgr

    cv2.imwrite("board.png", board)
    print("[SUCCESS] Generated canonical 512x512 board.png")


if __name__ == "__main__":
    main()