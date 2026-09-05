"""Generate frames across a realistic range, run the detector, and check
measured values against the known ground truth.

Exits non-zero if any case exceeds the error threshold.
"""

import math
import sys

import numpy as np

import generate
from generate import generate_frame
from detect import measure

SPEED_ERROR_THRESHOLD_PCT = 0.5
ANGLE_ERROR_THRESHOLD_DEG = 0.5

# (ball_speed_mph, launch_angle_deg, num_flashes) -- each case must fit
# fully in frame at generate.DEFAULT_STROBE_INTERVAL_MS with the default
# half-ball-diameter start margin, or generate_frame raises.
CASES = [
    (60, 8, 4),
    (86, 26, 4),   # wedge
    (100, 20, 5),
    (120, 15, 4),
    (150, 12, 4),  # driver
    (150, 12, 3),
    (180, 10, 4),
    (180, 35, 4),
    (60, 35, 6),
]


def run_sweep():
    print(f"{'speed(actual)':>14} {'speed(meas)':>12} {'err%':>7} | "
          f"{'angle(actual)':>14} {'angle(meas)':>12} {'err(deg)':>9} | flashes")
    print("-" * 90)

    worst_speed_err = 0.0
    worst_angle_err = 0.0
    any_failed_detection = False

    for speed_mph, angle_deg, num_flashes in CASES:
        image, truth = generate_frame(
            ball_speed_mph=speed_mph,
            launch_angle_deg=angle_deg,
            num_flashes=num_flashes,
        )
        result = measure(image, truth["strobe_interval_ms"])

        if result is None:
            print(f"{speed_mph:>14} {'DETECT FAIL':>12} {'':>7} | "
                  f"{angle_deg:>14} {'':>12} {'':>9} | {num_flashes}")
            any_failed_detection = True
            continue

        speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
        angle_err_deg = abs(result["launch_angle_deg"] - angle_deg)

        worst_speed_err = max(worst_speed_err, speed_err_pct)
        worst_angle_err = max(worst_angle_err, angle_err_deg)

        print(f"{speed_mph:>14.1f} {result['ball_speed_mph']:>12.2f} {speed_err_pct:>6.3f}% | "
              f"{angle_deg:>14.1f} {result['launch_angle_deg']:>12.2f} {angle_err_deg:>9.3f} | {num_flashes}")

    print("-" * 90)
    print(f"worst speed error: {worst_speed_err:.3f}% (threshold {SPEED_ERROR_THRESHOLD_PCT}%)")
    print(f"worst angle error: {worst_angle_err:.3f} deg (threshold {ANGLE_ERROR_THRESHOLD_DEG} deg)")

    passed = (
        not any_failed_detection
        and worst_speed_err <= SPEED_ERROR_THRESHOLD_PCT
        and worst_angle_err <= ANGLE_ERROR_THRESHOLD_DEG
    )
    print("PASS" if passed else "FAIL")
    return passed


def check_clipped_ball_exclusion():
    """A ball clipped by the frame edge must be dropped entirely, not
    dragged into the diameter average or the line fit."""
    speed_mph, angle_deg, num_flashes = 90.0, 15.0, 4
    interval_ms = generate.DEFAULT_STROBE_INTERVAL_MS
    radius_px = generate.BALL_DIAMETER_PX / 2.0

    speed_mps = speed_mph * generate.MPH_TO_MPS
    step_px = (speed_mps * 1000.0 * (interval_ms / 1000.0)) / generate.SCALE_MM_PER_PX
    dx_px = step_px * math.cos(math.radians(angle_deg))

    # Position the flight so the last ball's centre sits 30% of a diameter
    # inside the right edge -- clearly clipped, not off-frame.
    last_cx = generate.SENSOR_WIDTH_PX - 0.3 * (2 * radius_px)
    start_x = last_cx - (num_flashes - 1) * dx_px
    start_y = generate.SENSOR_HEIGHT_PX / 2.0

    image, truth = generate_frame(
        speed_mph, angle_deg, num_flashes=num_flashes,
        strobe_interval_ms=interval_ms, start_pos_px=(start_x, start_y),
    )
    statuses = [b["status"] for b in truth["balls"]]
    assert statuses.count("clipped") >= 1, f"test setup didn't produce a clipped ball: {statuses}"

    result = measure(image, interval_ms)
    assert result is not None, "detector failed on a shot with one clipped ball"

    speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
    print(f"clipped-ball case: {statuses.count('clipped')} clipped ball(s) in truth, "
          f"{result['excluded_border_blobs']} excluded by detector, "
          f"speed error {speed_err_pct:.3f}%")

    passed = (
        result["excluded_border_blobs"] >= 1
        and speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    )
    print("PASS" if passed else "FAIL")
    return passed


def check_nonzero_background():
    """A constant non-zero background must not bias area or centroid."""
    speed_mph, angle_deg, num_flashes = 100.0, 20.0, 5
    image, truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes)

    background_level = 25
    noisy = np.clip(image.astype(np.int16) + background_level, 0, 255).astype(np.uint8)

    result = measure(noisy, truth["strobe_interval_ms"])
    assert result is not None, "detector failed against a non-zero background"

    speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
    angle_err_deg = abs(result["launch_angle_deg"] - angle_deg)
    print(f"non-zero background (level {background_level}): "
          f"speed error {speed_err_pct:.3f}%, angle error {angle_err_deg:.3f} deg")

    passed = speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT and angle_err_deg <= ANGLE_ERROR_THRESHOLD_DEG
    print("PASS" if passed else "FAIL")
    return passed


def main():
    sweep_ok = run_sweep()
    print()
    clip_ok = check_clipped_ball_exclusion()
    print()
    background_ok = check_nonzero_background()

    if not (sweep_ok and clip_ok and background_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
