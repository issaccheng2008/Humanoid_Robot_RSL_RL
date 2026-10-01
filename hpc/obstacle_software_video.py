"""Draw an obstacle crossing side view without Kit, RTX, or a display server."""

from __future__ import annotations

import numpy as np


WIDTH = 960
HEIGHT = 544
GROUND_Z = 0.0
X_BEHIND = 0.65
X_AHEAD = 0.95
Z_BOTTOM = -0.08
Z_TOP = 0.58


class SoftwareVideoWriter:
    """Encode generated RGB frames through OpenCV's existing MP4 path."""

    def __init__(self, path: str, fps: float, width: int = WIDTH, height: int = HEIGHT):
        import cv2

        self._cv2 = cv2
        self._width = width
        self._height = height
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV could not open software video output: {path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._writer.release()

    def append_data(self, frame: np.ndarray) -> None:
        if frame.shape != (self._height, self._width, 3) or frame.dtype != np.uint8:
            raise ValueError("Software video frames must be RGB uint8 at the configured size")
        self._writer.write(np.ascontiguousarray(frame[..., ::-1]))


def _pixel(x: float, z: float, center_x: float) -> tuple[int, int]:
    px = int(round((x - center_x + X_BEHIND) * WIDTH / (X_BEHIND + X_AHEAD)))
    py = int(round((Z_TOP - z) * HEIGHT / (Z_TOP - Z_BOTTOM)))
    return px, py


def _disc(frame: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    x0, x1 = max(0, x - radius), min(frame.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(frame.shape[0], y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    frame[y0:y1, x0:x1][mask] = color


def _segment(
    frame: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 5,
) -> None:
    length = max(abs(second[0] - first[0]), abs(second[1] - first[1]), 1)
    for fraction in np.linspace(0.0, 1.0, length + 1):
        x = round(first[0] + fraction * (second[0] - first[0]))
        y = round(first[1] + fraction * (second[1] - first[1]))
        _disc(frame, x, y, radius, color)


def render_side_frame(
    root_position: np.ndarray,
    body_names: list[str],
    body_positions: np.ndarray,
    bar_position: np.ndarray,
    crossed: bool,
) -> np.ndarray:
    """Render measured body and bar positions as one RGB schematic frame."""
    root = np.asarray(root_position)
    bodies = np.asarray(body_positions)
    bar = np.asarray(bar_position)
    if root.shape != (3,) or bodies.shape != (len(body_names), 3) or bar.shape != (3,):
        raise ValueError("Expected root (3), bodies (N, 3), and bar (3) positions")

    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (238, 244, 249)
    center_x = float(root[0])
    ground_y = _pixel(center_x, GROUND_Z, center_x)[1]
    frame[ground_y : ground_y + 3, :] = (75, 83, 91)
    frame[ground_y + 3 :, :] = (214, 223, 218)

    # A bright bar remains legible despite its actual 2 cm physical width.
    if float(bar[2]) > -0.5:
        bar_x, bar_y = _pixel(float(bar[0]), float(bar[2]), center_x)
        x0, x1 = max(0, bar_x - 7), min(WIDTH, bar_x + 8)
        y0, y1 = max(0, bar_y - 6), min(HEIGHT, bar_y + 7)
        frame[y0:y1, x0:x1] = (214, 65, 42)

    names = {name: index for index, name in enumerate(body_names)}
    root_px = _pixel(float(root[0]), float(root[2]), center_x)
    for side, color in (("l", (44, 112, 214)), ("r", (35, 158, 121))):
        points = [root_px]
        for name in (
            f"{side}_leg_pitch_link",
            f"{side}_knee_pitch_link",
            f"{side}_ankle_pitch_link",
            f"{side}_ankle_roll_link",
        ):
            if name in names:
                position = bodies[names[name]]
                points.append(_pixel(float(position[0]), float(position[2]), center_x))
        for first, second in zip(points, points[1:]):
            _segment(frame, first, second, color)
        for point in points[1:]:
            _disc(frame, *point, 7, color)
    _disc(frame, *root_px, 12, (45, 53, 75))

    # A simple status light: amber while approaching, green once crossed.
    frame[22:48, WIDTH - 50 : WIDTH - 24] = (35, 158, 121) if crossed else (231, 171, 51)
    return frame
