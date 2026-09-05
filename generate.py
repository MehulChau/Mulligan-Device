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

# The fastest expected shot (a driver, ~150 mph) must fit its requested
# number of exposures inside the sensor width -- that's what sets this
# interval, not the other way around. With a half-ball-diameter start
# margin (the first ball tangent to the left edge), 2.2 ms fits 5 flashes
# of a 150 mph/12deg driver fully in frame; 2.9 ms only fits 4.
DEFAULT_STROBE_INTERVAL_MS = 2.2

MPH_TO_MPS = 0.44704

# A ball placed exactly tangent to the frame edge is indistinguishable, from
# the image alone, from one that's genuinely clipped -- both produce a blob
# bounding box flush against the border. A few pixels of buffer beyond the
# bare half-diameter minimum avoids that ambiguity without materially
# changing how many flashes fit (the driver case below has ~20px of slack).
BORDER_SAFETY_PX = 3.0

# Default distance (px) from the flight's start position back to the
# strobe LED, used by the IR falloff model below.
DEFAULT_FALLOFF_REFERENCE_PX = 300.0


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


def _draw_ball_smeared(image, cx, cy, radius, direction, smear_px, value=255):
    """Draw a ball smeared along `direction` by `smear_px` of travel during
    the flash's finite duration.

    Models the flash as a uniform sweep of the ball's centre over
    `smear_px` during the exposure: the time-integrated exposure is the
    average, over many sub-positions along the sweep, of the same
    per-pixel anti-aliased disk coverage `_draw_ball` already uses. A
    closed-form version of this (treating the disk edge as a hard cutoff
    at radius+0.5) was tried first and rejected: it inflates the *whole*
    shape isotropically by (radius+0.5)/radius, biasing even the
    perpendicular extent by ~2% instead of leaving it exact. Supersampling
    the validated static-disk ramp avoids that -- it reproduces the true
    R^2/4 perpendicular variance to ~0.001%.
    """
    if smear_px <= 1e-6:
        _draw_ball(image, cx, cy, radius, value)
        return

    h, w = image.shape
    dxu, dyu = direction
    half = smear_px / 2.0
    margin = radius + 2
    x0 = max(int(math.floor(cx - half - margin)), 0)
    x1 = min(int(math.ceil(cx + half + margin)), w)
    y0 = max(int(math.floor(cy - half - margin)), 0)
    y1 = min(int(math.ceil(cy + half + margin)), h)
    if x0 >= x1 or y0 >= y1:
        return

    # Sub-positions spaced well under 1px apart so the average reproduces
    # the continuous time integral to sub-0.1% accuracy.
    n_samples = max(int(math.ceil(smear_px * 4.0)) + 1, 5)
    taus = np.linspace(-half, half, n_samples)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    accum = np.zeros(xs.shape, dtype=np.float64)
    for t in taus:
        sx, sy = cx + dxu * t, cy + dyu * t
        dist = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
        accum += np.clip(radius - dist + 0.5, 0.0, 1.0)
    coverage = np.clip(accum / n_samples, 0.0, 1.0)

    patch = (coverage * value).astype(np.uint8)
    image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], patch)


def _ball_visibility(cx, cy, radius, width, height):
    """Classify a ball image against the frame bounds: fully_in_frame,
    clipped (partially overlaps), or off_frame (no overlap, nothing drawn)."""
    left, right = cx - radius, cx + radius
    top, bottom = cy - radius, cy + radius
    if right <= 0 or left >= width or bottom <= 0 or top >= height:
        return "off_frame"
    if left < 0 or right > width or top < 0 or bottom > height:
        return "clipped"
    return "fully_in_frame"


def generate_frame(
    ball_speed_mph,
    launch_angle_deg,
    num_flashes=4,
    strobe_interval_ms=DEFAULT_STROBE_INTERVAL_MS,
    start_pos_px=None,
    width=SENSOR_WIDTH_PX,
    height=SENSOR_HEIGHT_PX,
    noise_sigma=0.0,
    flash_duration_us=0.0,
    ir_falloff=False,
    strobe_source_px=None,
    falloff_reference_px=DEFAULT_FALLOFF_REFERENCE_PX,
):
    """Draw a grayscale frame with `num_flashes` ball images along a line.

    noise_sigma: stddev (DN) of additive Gaussian sensor noise, applied
        after everything else. 0 (default) draws a clean, noiseless frame.
    flash_duration_us: how long the strobe stays on per flash. 0 (default)
        draws an instantaneous point-in-time ball (no motion smear); above
        0, each ball is smeared along its direction of travel to represent
        how far it moves during the flash.
    ir_falloff: if True, each ball's brightness follows inverse-square
        falloff from a point-source LED at `strobe_source_px` (default:
        `falloff_reference_px` behind the flight's start, along the line
        of travel), normalized so the nearest ball is at full brightness.

    Returns (image, ground_truth_dict).
    """
    image = np.zeros((height, width), dtype=np.uint8)

    speed_mps = ball_speed_mph * MPH_TO_MPS
    speed_px_per_s = (speed_mps * 1000.0) / SCALE_MM_PER_PX
    speed_mm_per_flash = speed_mps * 1000.0 * (strobe_interval_ms / 1000.0)
    step_px = speed_mm_per_flash / SCALE_MM_PER_PX

    angle_rad = math.radians(launch_angle_deg)
    direction = (math.cos(angle_rad), -math.sin(angle_rad))  # y flips: image y increases downward
    dx_px = step_px * direction[0]
    dy_px = step_px * direction[1]

    radius_px = BALL_DIAMETER_PX / 2.0

    if start_pos_px is None:
        # Half a ball diameter from the edge, plus a small border-safety
        # buffer (see BORDER_SAFETY_PX above).
        margin = radius_px + BORDER_SAFETY_PX
        start_x = margin
        start_y = height - margin
        start_pos_px = (start_x, start_y)

    centers = []
    statuses = []
    for i in range(num_flashes):
        cx = start_pos_px[0] + i * dx_px
        cy = start_pos_px[1] + i * dy_px
        centers.append((cx, cy))
        statuses.append(_ball_visibility(cx, cy, radius_px, width, height))

    off_frame_count = statuses.count("off_frame")
    if off_frame_count:
        raise ValueError(
            f"{off_frame_count} of {num_flashes} requested flashes fall entirely "
            "outside the frame at this speed/angle/interval -- the requested "
            "scenario was not produced. Reduce num_flashes, shorten "
            "strobe_interval_ms, or move start_pos_px."
        )

    if ir_falloff:
        if strobe_source_px is None:
            strobe_source_px = (start_pos_px[0] - falloff_reference_px, start_pos_px[1])
        distances = [math.hypot(cx - strobe_source_px[0], cy - strobe_source_px[1]) for cx, cy in centers]
        min_distance = min(distances)
        intensities = [255.0 * (min_distance / d) ** 2 for d in distances]
    else:
        strobe_source_px = None
        intensities = [255.0] * num_flashes

    smear_px = speed_px_per_s * (flash_duration_us * 1e-6) if flash_duration_us > 0 else 0.0

    balls = []
    for (cx, cy), status, intensity in zip(centers, statuses, intensities):
        balls.append({"center_px": (cx, cy), "status": status, "intensity": intensity})
        if status == "off_frame":
            continue  # nothing to draw -- it isn't in the image at all
        value = int(round(min(max(intensity, 0.0), 255.0)))
        if smear_px > 0:
            _draw_ball_smeared(image, cx, cy, radius_px, direction, smear_px, value=value)
        else:
            _draw_ball(image, cx, cy, radius_px, value=value)

    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0.0, 255.0).astype(np.uint8)

    ground_truth = {
        "ball_speed_mph": ball_speed_mph,
        "launch_angle_deg": launch_angle_deg,
        "num_flashes": num_flashes,
        "strobe_interval_ms": strobe_interval_ms,
        "balls": balls,
        "ball_diameter_px": radius_px * 2.0,
        "scale_mm_per_px": SCALE_MM_PER_PX,
        "image_width": width,
        "image_height": height,
        "noise_sigma": noise_sigma,
        "flash_duration_us": flash_duration_us,
        "smear_px": smear_px,
        "ir_falloff": ir_falloff,
        "strobe_source_px": strobe_source_px,
    }
    return image, ground_truth


def save_frame(image, ground_truth, image_path, json_path):
    cv2.imwrite(image_path, image)
    with open(json_path, "w") as f:
        json.dump(ground_truth, f, indent=2)


if __name__ == "__main__":
    # A 150 mph driver at 12deg fits 4 flashes fully in frame at the
    # default 2.2 ms interval (5 fits exactly at the boundary -- 4 leaves
    # headroom for a real, non-idealized shot).
    img, gt = generate_frame(ball_speed_mph=150.0, launch_angle_deg=12.0, num_flashes=4)
    save_frame(img, gt, "frame.png", "frame.json")
    print(f"scale: {SCALE_MM_PER_PX:.4f} mm/px, ball diameter: {BALL_DIAMETER_PX:.1f} px")
    print("wrote frame.png and frame.json")
