"""Generate frames across a realistic range, run the detector, and check
measured values against the known ground truth.

Exits non-zero if any case exceeds the error threshold.
"""

import sys

from generate import generate_frame
from detect import measure

SPEED_ERROR_THRESHOLD_PCT = 0.5
ANGLE_ERROR_THRESHOLD_DEG = 0.5

CASES = [
    # (ball_speed_mph, launch_angle_deg, num_flashes)
    (60, 8, 4),
    (86, 26, 4),   # wedge
    (100, 20, 5),
    (120, 15, 4),
    (150, 12, 5),  # driver
    (150, 12, 3),
    (180, 10, 6),
    (180, 35, 4),
    (60, 35, 6),
]


def main():
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

    failed = (
        any_failed_detection
        or worst_speed_err > SPEED_ERROR_THRESHOLD_PCT
        or worst_angle_err > ANGLE_ERROR_THRESHOLD_DEG
    )
    if failed:
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
