"""Generate synthetic strobe photographs of a golf ball in flight.

Simulates what the camera will produce: several IR-strobe flashes fire in
quick succession while the shutter stays open, so one frame contains
several images of the same ball along a rising line. Over the ~10 ms a
frame covers, gravity and drag move the ball a fraction of a millimeter
compared to its horizontal travel, so the flight is modeled as a straight
line at constant speed.
"""

import json
import math

import cv2
import numpy as np

# --- Camera / geometry constants (planned hardware, tune later) ---
SENSOR_WIDTH_PX = 1456
SENSOR_HEIGHT_PX = 1088
LENS_FOCAL_LENGTH_MM = 6.0
SENSOR_WIDTH_MM = 6.3
CAMERA_DISTANCE_M = 0.6

# Field of view and scale derived from the pinhole camera model:
# fov_width = sensor_width_mm * distance / focal_length
FOV_WIDTH_M = SENSOR_WIDTH_MM * CAMERA_DISTANCE_M / LENS_FOCAL_LENGTH_MM
SCALE_MM_PER_PX = (FOV_WIDTH_M * 1000.0) / SENSOR_WIDTH_PX

BALL_DIAMETER_MM = 42.67
BALL_DIAMETER_PX = BALL_DIAMETER_MM / SCALE_MM_PER_PX

DEFAULT_STROBE_INTERVAL_MS = 2.9

MPH_TO_MPS = 0.44704


def _draw_ball(image, cx, cy, radius, value=255):
    """Draw a filled disk with a 1px anti-aliased edge at a sub-pixel centre.

    A real lens never produces a perfectly hard-edged ball image, so an
    anti-aliased edge is the physically honest choice, not just cosmetic.
    It also matters for testing: cv2.circle's own rasterization (even with
    its sub-pixel `shift` option) inflates the apparent radius by roughly a
    constant number of pixels regardless of threshold, which is a large
    relative error at this ball's ~99px scale. Computing per-pixel coverage
    directly from the true distance to the sub-pixel centre avoids that and
    keeps the drawn disk within ~0.01% of the requested radius.
    """
    h, w = image.shape
    margin = radius + 2
    x0 = max(int(math.floor(cx - margin)), 0)
    x1 = min(int(math.ceil(cx + margin)), w)
    y0 = max(int(math.floor(cy - margin)), 0)
    y1 = min(int(math.ceil(cy + margin)), h)
    if x0 >= x1 or y0 >= y1:
        return

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    # Fully lit inside (radius - 0.5), fully dark outside (radius + 0.5),
    # linear ramp across the 1px boundary in between.
    coverage = np.clip(radius - dist + 0.5, 0.0, 1.0)
    patch = (coverage * value).astype(np.uint8)
    image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], patch)


def generate_frame(
    ball_speed_mph,
    launch_angle_deg,
    num_flashes=4,
    strobe_interval_ms=DEFAULT_STROBE_INTERVAL_MS,
    start_pos_px=None,
    width=SENSOR_WIDTH_PX,
    height=SENSOR_HEIGHT_PX,
):
    """Draw a grayscale frame with `num_flashes` ball images along a line.

    Returns (image, ground_truth_dict).
    """
    image = np.zeros((height, width), dtype=np.uint8)

    speed_mps = ball_speed_mph * MPH_TO_MPS
    speed_mm_per_flash = speed_mps * 1000.0 * (strobe_interval_ms / 1000.0)
    step_px = speed_mm_per_flash / SCALE_MM_PER_PX

    angle_rad = math.radians(launch_angle_deg)
    dx_px = step_px * math.cos(angle_rad)
    # Image y increases downward; the ball rises, so y decreases.
    dy_px = -step_px * math.sin(angle_rad)

    if start_pos_px is None:
        # Start near the bottom-left, leaving room for the full flight path.
        margin = BALL_DIAMETER_PX
        start_x = margin * 1.5
        start_y = height - margin * 1.5
        start_pos_px = (start_x, start_y)

    radius_px = BALL_DIAMETER_PX / 2.0
    centers = []
    for i in range(num_flashes):
        cx = start_pos_px[0] + i * dx_px
        cy = start_pos_px[1] + i * dy_px
        centers.append((cx, cy))
        _draw_ball(image, cx, cy, radius_px)

    ground_truth = {
        "ball_speed_mph": ball_speed_mph,
        "launch_angle_deg": launch_angle_deg,
        "num_flashes": num_flashes,
        "strobe_interval_ms": strobe_interval_ms,
        "ball_centers_px": centers,
        "ball_diameter_px": radius_px * 2.0,
        "scale_mm_per_px": SCALE_MM_PER_PX,
        "image_width": width,
        "image_height": height,
    }
    return image, ground_truth


def save_frame(image, ground_truth, image_path, json_path):
    cv2.imwrite(image_path, image)
    with open(json_path, "w") as f:
        json.dump(ground_truth, f, indent=2)


if __name__ == "__main__":
    img, gt = generate_frame(ball_speed_mph=150.0, launch_angle_deg=12.0, num_flashes=5)
    save_frame(img, gt, "frame.png", "frame.json")
    print(f"scale: {SCALE_MM_PER_PX:.4f} mm/px, ball diameter: {BALL_DIAMETER_PX:.1f} px")
    print("wrote frame.png and frame.json")
