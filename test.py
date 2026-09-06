"""Generate frames across a realistic range, run the detector, and check
measured values against the known ground truth.

Exits non-zero if any case exceeds the error threshold.
"""

import math
import os
import sys
import tempfile

import cv2
import numpy as np

import calibrate
import generate
from generate import generate_frame
import detect
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


BACKGROUND_LEVEL = 25
NOISE_TRIALS = 15
NOISE_LEVELS = (0.0, 3.0, 8.0, 15.0, 25.0)


def _apply_background_and_noise(image, background, sigma):
    """Add a constant background and Gaussian noise *together*, background
    first. Adding noise inside generate_frame (background=0, clipped there)
    and only afterward adding a background offset double-clips: negative
    noise excursions get rectified against a floor of 0 before the
    background ever gets a chance to give them room, producing a
    degenerate (near-zero) noise estimate downstream. Establishing the true
    background first means noise can push values symmetrically above and
    below it, as it would in a real image."""
    noisy = image.astype(np.float64) + background
    if sigma > 0:
        noisy = noisy + np.random.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def check_noise_sweep():
    """Gating: background + Gaussian noise must not push speed error past
    threshold at sigma = 0, 3, 8, 15, 25. This is what would have caught
    the rectified-moment bias (detect.py rectifying the background-
    subtracted residual at zero, which one-sidedly inflated area/variance)
    and the fixed-halving adaptive threshold dipping below background --
    both are now fixed, but this sweep is what should catch a regression
    of either one, so it gates instead of just reporting."""
    np.random.seed(20240605)  # fixed seed: a stable gate, not a flaky one
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    passed = True
    print(f"noise sweep (background level {BACKGROUND_LEVEL}, {NOISE_TRIALS} trials/level, mean-gated):")
    for sigma in NOISE_LEVELS:
        errs = []
        fails = 0
        for _ in range(NOISE_TRIALS):
            image, truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes)
            noisy = _apply_background_and_noise(image, BACKGROUND_LEVEL, sigma)
            result = measure(noisy, truth["strobe_interval_ms"], threshold=40, adaptive=True)
            if result is None:
                fails += 1
                continue
            errs.append(abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0)
        if errs:
            mean_err = float(np.mean(errs))
            print(f"  sigma={sigma:5.1f}: mean err={mean_err:.3f}%  "
                  f"max err={np.max(errs):.3f}%  detect fails={fails}/{NOISE_TRIALS}")
            passed &= fails == 0 and mean_err <= SPEED_ERROR_THRESHOLD_PCT
        else:
            print(f"  sigma={sigma:5.1f}: all {NOISE_TRIALS} detections failed")
            passed = False

    print("PASS" if passed else "FAIL")
    return passed


def check_motion_smear():
    """Motion smear stretches a blob along the direction of travel, biasing
    an area-based diameter estimate. Deriving scale from each blob's extent
    perpendicular to the fitted flight line should be immune to it."""
    passed = True
    for name, speed_mph, angle_deg, num_flashes in (("driver", 150.0, 12.0, 4), ("wedge", 86.0, 26.0, 4)):
        flash_duration_us = 30.0
        image, truth = generate_frame(
            speed_mph, angle_deg, num_flashes=num_flashes, flash_duration_us=flash_duration_us,
        )
        result = measure(image, truth["strobe_interval_ms"])
        assert result is not None, f"detector failed on a smeared {name} shot"

        speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
        print(f"motion smear ({name}, {truth['smear_px']:.2f}px @ {flash_duration_us}us): "
              f"speed error {speed_err_pct:.4f}%")
        passed &= speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT

    print("PASS" if passed else "FAIL")
    return passed


def check_ir_falloff():
    """IR falloff dims far balls enough that a single global threshold
    misses them outright. Report that failure, then confirm the adaptive
    (multi-threshold) detector recovers."""
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    image, truth = generate_frame(
        speed_mph, angle_deg, num_flashes=num_flashes, ir_falloff=True,
    )
    intensities = [b["intensity"] for b in truth["balls"]]
    print(f"IR falloff ball intensities (0-255): {[f'{v:.1f}' for v in intensities]}")

    global_result = measure(image, truth["strobe_interval_ms"], threshold=40, adaptive=False)
    if global_result is None:
        print("global threshold=40: detection FAILED (fewer than 3 ball images found), as expected")
    else:
        print(f"global threshold=40: detection unexpectedly succeeded "
              f"({global_result['num_ball_images']} blobs)")

    adaptive_result = measure(image, truth["strobe_interval_ms"], threshold=40, adaptive=True)
    assert adaptive_result is not None, "adaptive detector failed to recover under IR falloff"
    speed_err_pct = abs(adaptive_result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
    print(f"adaptive threshold: {adaptive_result['num_ball_images']} blobs found, "
          f"speed error {speed_err_pct:.3f}%")

    passed = speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


def check_falloff_with_background_noise():
    """Falloff + background + noise together -- what a real range frame
    actually looks like, and a combination neither the falloff test (clean
    background) nor the noise sweep (uniform brightness) covers alone. The
    specific regression this guards: a fixed halving threshold sequence
    (40, 20, 10, 5) steps below a background of 25, thresholding the
    *entire* frame to white -- one contour that can even pass the
    circularity filter. The derived-from-statistics sequence must never
    dip at or below the frame's own measured background."""
    np.random.seed(20240605)
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    noise_sigma = 5.0  # modest but real; see report for how far this can be pushed

    image, truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes, ir_falloff=True)
    noisy = _apply_background_and_noise(image, BACKGROUND_LEVEL, noise_sigma)

    background_est = float(np.median(noisy))
    thresholds = detect._adaptive_thresholds(noisy, 40)
    print(f"falloff + background({BACKGROUND_LEVEL}) + noise(sigma={noise_sigma}): "
          f"estimated background={background_est:.1f}, thresholds={[f'{t:.1f}' for t in thresholds]}")
    no_dip = all(t > background_est for t in thresholds)

    result = measure(noisy, truth["strobe_interval_ms"], threshold=40, adaptive=True)
    detected = result is not None and result["num_ball_images"] >= MIN_BALL_IMAGES_EXPECTED
    speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0 if result else float("nan")
    print(f"  {result['num_ball_images'] if result else 0} blobs found, speed error {speed_err_pct:.3f}%")

    passed = no_dip and detected and speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


MIN_BALL_IMAGES_EXPECTED = 3


REAL_BALL_AREA_RANGE = (7000.0, 8300.0)  # a real ball image's contour area, established empirically


def _contour_survives_filters(image):
    """Re-run the same area/circularity/border logic detect.py's blob
    finder uses, over every contour that isn't a real ball's, and return
    info for the largest one (the distractor, assuming just one is
    present) -- or None if nothing but real balls was found."""
    _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_px = 0.05 * image.shape[0] * image.shape[1]
    candidates = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 5 or REAL_BALL_AREA_RANGE[0] <= a <= REAL_BALL_AREA_RANGE[1]:
            continue  # skip noise specks and real ball contours
        perimeter = cv2.arcLength(c, True)
        circ = 4 * math.pi * a / (perimeter * perimeter) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(c)
        touches_border = x <= 0 or y <= 0 or x + w >= image.shape[1] or y + h >= image.shape[0]
        area_ok = 20 <= a <= max_area_px
        circ_ok = circ >= 0.7
        candidates.append({"area": a, "circ": circ, "area_ok": area_ok, "circ_ok": circ_ok,
                            "border": touches_border, "survives": area_ok and circ_ok and not touches_border})
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["area"])


def check_distractors():
    """Report (not gate, except where a real defense exists) how each
    distractor fares against detect.py's *basic* per-blob filters --
    area, circularity, and border-touching -- in isolation. The laser dot
    and an isolated round reflection are known to survive these alone;
    what actually rejects them is the size-consistency filter and the
    RANSAC line fit in measure(), exercised end-to-end (with all three
    distractors, plus noise and falloff, at once) by
    check_distractors_combined below."""
    print("distractor survival (basic per-blob filters only):")
    passed = True

    # Laser dot: round and bright like a real ball, just much smaller.
    # Sweep radius to find where it starts clearing the area filter.
    for r in (2.0, 2.5, 3.0, 4.0, 8.0):
        image, truth = generate_frame(150.0, 12.0, num_flashes=4, laser_dot=True, laser_dot_radius_px=r)
        approx_area = math.pi * r * r
        info = _contour_survives_filters(image)
        status = info["survives"] if info else False
        print(f"  laser dot r={r:4.1f}px (~{approx_area:6.1f}px^2): "
              f"{'SURVIVES filters (false positive)' if status else 'rejected'}"
              + (f"  [area_ok={info['area_ok']} circ={info['circ']:.2f}]" if info else " [no separate contour]"))

    # An 8px laser dot survives the basic filters above, but full measure()
    # also applies the size-consistency filter and RANSAC line fit -- this
    # confirms *those* are what actually keep the measurement clean, not
    # area/circularity (which is exactly why check_distractors_combined
    # gates on the full pipeline instead of this basic-filter view).
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    clean_image, clean_truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes)
    clean_result = measure(clean_image, clean_truth["strobe_interval_ms"])
    laser_image, laser_truth = generate_frame(
        speed_mph, angle_deg, num_flashes=num_flashes, laser_dot=True, laser_dot_radius_px=8.0,
    )
    laser_result = measure(laser_image, laser_truth["strobe_interval_ms"])
    if laser_result is not None:
        err = abs(laser_result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
        print(f"  full pipeline with an 8px laser dot present: {laser_result['num_ball_images']} blobs used "
              f"(vs {clean_result['num_ball_images']} clean), excluded_size={laser_result['excluded_size_blobs']}, "
              f"speed error {err:.3f}% "
              f"(area/circularity alone would have let this through -- size-consistency filtering "
              f"and the RANSAC line fit in measure() are the actual defense)")

    # Mat reflection: diffuse, but the default is *elongated* -- that's what
    # circularity actually catches here, not the soft edge. An isolated,
    # round diffuse reflection is checked separately to make that explicit.
    image, truth = generate_frame(150.0, 12.0, num_flashes=4, mat_reflection=True)
    info = _contour_survives_filters(image)
    print(f"  mat reflection (elongated, default): "
          f"{'SURVIVES' if info and info['survives'] else 'rejected'}"
          + (f"  [circ={info['circ']:.2f}]" if info else ""))
    passed &= bool(info) and not info["survives"]

    from generate import _draw_mat_reflection
    image_iso, _ = generate_frame(150.0, 12.0, num_flashes=4)
    _draw_mat_reflection(image_iso, 700.0, 300.0, sigma_x=45.0, sigma_y=45.0, peak=70.0)  # isolated, round
    info_iso = _contour_survives_filters(image_iso)
    print(f"  mat reflection (round, isolated): "
          f"{'SURVIVES filters (roundness alone is not sharpness)' if info_iso and info_iso['survives'] else 'rejected'}"
          + (f"  [circ={info_iso['circ']:.2f}]" if info_iso else ""))

    # Clubhead edge: bright and hard-edged, but triangular and
    # border-touching by construction -- expect both filters to catch it.
    image, truth = generate_frame(150.0, 12.0, num_flashes=4, clubhead_edge=True)
    info = _contour_survives_filters(image)
    print(f"  clubhead edge: {'SURVIVES' if info and info['survives'] else 'rejected'}"
          + (f"  [circ={info['circ']:.2f}, border={info['border']}]" if info else ""))
    passed &= bool(info) and not info["survives"]

    print("PASS" if passed else "FAIL")
    return passed


def check_distractors_combined():
    """Gating: laser dot, mat reflection, and clubhead edge all present at
    once, plus noise and IR falloff -- the realistic worst case. The
    size-consistency filter (item 1) and RANSAC line fit (item 2) are what
    should make this survivable now, where basic per-blob filtering alone
    could not (see check_distractors)."""
    np.random.seed(20240605)
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    noise_sigma = 3.0

    # The reflection's default position is close enough to a real ball
    # that, once IR falloff forces a very low adaptive threshold to see
    # the dim far balls, it merges into that ball's contour instead of
    # staying a separate (and separately rejectable) blob -- so it's
    # placed further from the flight line here, same as the isolated
    # roundness check above.
    image, truth = generate_frame(
        speed_mph, angle_deg, num_flashes=num_flashes,
        ir_falloff=True, laser_dot=True, laser_dot_radius_px=8.0,
        mat_reflection=True, mat_reflection_pos_px=(700.0, 300.0),
        clubhead_edge=True,
    )
    noisy = _apply_background_and_noise(image, BACKGROUND_LEVEL, noise_sigma)
    result = measure(noisy, truth["strobe_interval_ms"], threshold=40, adaptive=True)

    if result is None:
        print("distractors + noise + falloff combined: detection FAILED")
        print("FAIL")
        return False

    speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
    print(f"distractors + noise + falloff combined: {result['num_ball_images']} blobs used, "
          f"consensus {result['consensus_size']} "
          f"(excluded: border={result['excluded_border_blobs']}, "
          f"size={result['excluded_size_blobs']}, outlier={result['excluded_outlier_blobs']}), "
          f"speed error {speed_err_pct:.3f}%")

    passed = speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


def check_online_distractor_rejected():
    """The specific gap RANSAC's spacing check exists to close: a
    distractor sitting *on* the flight line (not off to the side) at an
    arbitrary offset -- e.g. a real laser dot marking the address
    position, which sits on the line rather than beside it like the
    generated one does. Collinearity alone can't reject this; only
    checking that its position breaks the regular flash-interval spacing
    can. Ball-sized on purpose, so the size filter can't be what saves it
    either -- this isolates the spacing check specifically."""
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    radius_px = generate.BALL_DIAMETER_PX / 2.0
    start_pos = (400.0, 900.0)  # clear of the left edge, room for a distractor behind it
    image, truth = generate_frame(
        speed_mph, angle_deg, num_flashes=num_flashes, start_pos_px=start_pos,
    )

    angle_rad = math.radians(angle_deg)
    direction = (math.cos(angle_rad), -math.sin(angle_rad))
    # 180px behind ball1, on the extended flight line -- collinear, and
    # its ratio to the real ~333px spacing (about 1.85x) is deliberately
    # close to 2x to stress-test the spacing check against the missing-
    # flash allowance it has to coexist with.
    distractor_pos = (start_pos[0] - 180 * direction[0], start_pos[1] - 180 * direction[1])
    generate._draw_ball(image, distractor_pos[0], distractor_pos[1], radius_px, value=255)

    result = measure(image, truth["strobe_interval_ms"])
    assert result is not None, "detector failed on a shot with an on-line ball-sized distractor"

    speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
    print(f"on-line distractor (ball-sized, non-flash-timed offset): "
          f"{result['num_ball_images']} blobs used, excluded_outlier={result['excluded_outlier_blobs']}, "
          f"speed error {speed_err_pct:.4f}%")

    passed = result["excluded_outlier_blobs"] >= 1 and speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


def check_missing_flash():
    """A missing/misfired exposure leaves one gap at ~2x width. Using mean
    spacing (before the fix) reads speed high; the fix should recover the
    true speed via the minimum consistent spacing instead."""
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 5
    image, truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes, missing_flash_index=2)
    result = measure(image, truth["strobe_interval_ms"])
    assert result is not None, "detector failed on a shot with one missing flash"

    speed_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0
    print(f"missing flash (index 2 of 5): speed error {speed_err_pct:.3f}%, "
          f"spacing_consistency={result['spacing_consistency']:.3f}, "
          f"missing_flash_detected={result['missing_flash_detected']}")

    passed = result["missing_flash_detected"] and speed_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


CALIBRATION_TEST_POSES = [
    # (board_distance_m, board_tilt_deg, board_offset_mm) -- varied *angle*
    # is what actually constrains the fit; fronto-parallel photos at only
    # varied positions/scales under-constrain it (verified empirically:
    # it recovers a self-consistent but badly wrong camera matrix with no
    # warning -- focal length and distortion trade off against each other
    # without real perspective to break the tie).
    (0.5, (0, 0), (0, 0)), (0.5, (25, 0), (0, 0)), (0.5, (-25, 0), (0, 0)),
    (0.5, (0, 25), (0, 0)), (0.5, (0, -25), (0, 0)),
    (0.45, (20, 20), (-100, -80)), (0.45, (20, -20), (100, -80)),
    (0.45, (-20, 20), (-100, 80)), (0.45, (-20, -20), (100, 80)),
    (0.6, (15, 15), (150, 100)), (0.55, (10, -10), (-150, 100)),
    (0.35, (0, 0), (0, 0)), (0.7, (0, 0), (0, 0)),
]


def check_calibration_recovers_distortion():
    """Generate synthetic checkerboards at varied poses, run calibrate.py's
    pipeline against them, and confirm it recovers the known distortion
    coefficients and focal length."""
    true_dist = (-0.15, 0.05, 0.0, 0.0, 0.0)
    pattern_size = (9, 6)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, (dist_m, tilt, offset) in enumerate(CALIBRATION_TEST_POSES):
            img, _ = generate.generate_checkerboard_frame(
                pattern_size=pattern_size, board_distance_m=dist_m,
                board_tilt_deg=tilt, board_offset_mm=offset, dist_coeffs=true_dist,
            )
            path = os.path.join(tmpdir, f"board_{i:02d}.png")
            cv2.imwrite(path, img)
            image_paths.append(path)

        result = calibrate.calibrate(image_paths, pattern_size, square_size_mm=25.0,
                                      max_reprojection_error_px=1.0)

    recovered_dist = result["dist_coeffs"]
    k1_err = abs(recovered_dist[0] - true_dist[0])
    k2_err = abs(recovered_dist[1] - true_dist[1])
    fx_err_pct = abs(result["camera_matrix"][0][0] - generate.FOCAL_LENGTH_PX) / generate.FOCAL_LENGTH_PX * 100.0

    print(f"calibration recovery: reprojection error {result['reprojection_error_px']:.4f}px, "
          f"k1={recovered_dist[0]:.4f} (true {true_dist[0]}), "
          f"k2={recovered_dist[1]:.4f} (true {true_dist[1]}), "
          f"focal length error {fx_err_pct:.3f}%")

    passed = k1_err < 0.02 and k2_err < 0.03 and fx_err_pct < 1.0
    print("PASS" if passed else "FAIL")
    return passed


def check_undistortion_reduces_speed_error():
    """A ball frame shot through a distorted lens should show measurable
    speed error; undistorting it with the true calibration should reduce
    that error. k1=-0.10 corresponds to roughly 3% displacement at the
    frame corner -- a realistic cheap-lens barrel distortion level, not a
    worst case (see calibrate.py's own edge_shift_px)."""
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    true_dist = (-0.10, 0.0, 0.0, 0.0, 0.0)
    start_pos = (400.0, 850.0)  # flight ends near the right edge, where distortion matters most

    image_dist, truth_dist = generate_frame(
        speed_mph, angle_deg, num_flashes=num_flashes, start_pos_px=start_pos, dist_coeffs=true_dist,
    )
    uncorrected = measure(image_dist, truth_dist["strobe_interval_ms"])
    assert uncorrected is not None, "detector failed on a distorted frame"
    uncorrected_err = abs(uncorrected["ball_speed_mph"] - speed_mph) / speed_mph * 100.0

    calibration = {"camera_matrix": generate.DEFAULT_CAMERA_MATRIX, "dist_coeffs": list(true_dist)}
    corrected = measure(image_dist, truth_dist["strobe_interval_ms"], calibration=calibration)
    assert corrected is not None, "detector failed on the undistorted frame"
    corrected_err = abs(corrected["ball_speed_mph"] - speed_mph) / speed_mph * 100.0

    print(f"barrel distortion (k1={true_dist[0]}): uncorrected speed error {uncorrected_err:.3f}%, "
          f"corrected {corrected_err:.4f}%, calibration_applied={corrected['calibration_applied']}")

    passed = corrected_err < uncorrected_err and corrected_err <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


def check_per_blob_scale_depth_varying():
    """A flight where the ball's distance from the camera changes (moving
    toward or away, not just laterally), so its apparent diameter changes
    across the frame -- exactly what a real 3D trajectory does that this
    project's flat-scale model otherwise ignores. Deriving scale per blob
    and averaging the two endpoint scales for each gap should measure
    this correctly; a single global mean scale, applied everywhere,
    should not.

    ~9% depth change by the last flash is a realistic upper bound for
    this camera's geometry (2 feet to the side, most of the flight
    parallel to the sensor) -- and conveniently, also about the most the
    RANSAC line fit's spacing tolerance can absorb before it starts
    finding the depth-varying (but perfectly genuine) pixel spacing
    inconsistent with a single flash interval; see
    docs/hardware-readiness.md.
    """
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    depth_profile = [1.0, 1.03, 1.06, 1.09]  # ball moving away across the frame
    image, truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes, depth_profile=depth_profile)

    result = measure(image, truth["strobe_interval_ms"])
    assert result is not None, "detector failed on a depth-varying flight"
    new_err_pct = abs(result["ball_speed_mph"] - speed_mph) / speed_mph * 100.0

    # Reconstruct what a single-global-mean-scale approach would have
    # given, for comparison: one scale from the mean diameter, applied to
    # the mean pixel spacing -- measure()'s behavior before per-blob
    # scale replaced it.
    old_scale = generate.BALL_DIAMETER_MM / result["mean_diameter_px"]
    centers = np.array(result["ball_centers"])
    pixel_spacings = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    old_speed_mps = (float(np.mean(pixel_spacings)) * old_scale / 1000.0) / (truth["strobe_interval_ms"] / 1000.0)
    old_speed_mph = old_speed_mps / generate.MPH_TO_MPS
    old_err_pct = abs(old_speed_mph - speed_mph) / speed_mph * 100.0

    print(f"depth-varying flight (per-blob scale {result['per_blob_scales_mm_per_px'][0]:.4f} -> "
          f"{result['per_blob_scales_mm_per_px'][-1]:.4f} mm/px): "
          f"mean-scale err={old_err_pct:.4f}%, per-blob-scale err={new_err_pct:.4f}%")

    passed = new_err_pct < old_err_pct and new_err_pct <= SPEED_ERROR_THRESHOLD_PCT
    print("PASS" if passed else "FAIL")
    return passed


def check_missing_flash_low_confidence_with_3_balls():
    """With only 3 balls (2 gaps), spacing_consistency is a noisy
    statistic and the minimum-of-2 is itself biased low -- the min-spacing
    correction must not fire here. The frame should be flagged
    low-confidence instead of "corrected" with a still-wrong number."""
    speed_mph, angle_deg, num_flashes = 150.0, 12.0, 4
    image, truth = generate_frame(speed_mph, angle_deg, num_flashes=num_flashes, missing_flash_index=1)
    result = measure(image, truth["strobe_interval_ms"])
    assert result is not None, "detector failed on a 3-ball shot with one missing flash"

    print(f"missing flash, 3 balls remaining: speed={result['ball_speed_mph']:.1f} mph "
          f"(true {speed_mph}), missing_flash_detected={result['missing_flash_detected']}, "
          f"low_confidence_spacing={result['low_confidence_spacing']}")

    passed = result["low_confidence_spacing"] and not result["missing_flash_detected"]
    print("PASS" if passed else "FAIL")
    return passed


def check_overlapping_balls_fail_safely():
    """A slow shot with a too-short interval overlaps consecutive ball
    images into one merged contour. The detector should fail safely
    (return None) rather than produce a corrupted number."""
    speed_mph, angle_deg = 60.0, 8.0
    # spacing ~= 0.7x ball diameter: comfortably overlapping
    image, truth = generate_frame(speed_mph, angle_deg, num_flashes=4, strobe_interval_ms=1.114)
    result = measure(image, truth["strobe_interval_ms"])
    print(f"overlapping balls (spacing ~0.7x diameter): "
          f"{'correctly returned None' if result is None else f'UNSAFE: returned {result}'}")

    passed = result is None
    print("PASS" if passed else "FAIL")
    return passed


def check_no_ball_frame():
    """A frame with no ball at all must fail safely, not crash or guess."""
    image, truth = generate_frame(150.0, 12.0, num_flashes=0)
    result = measure(image, truth["strobe_interval_ms"])
    print(f"no-ball frame: {'correctly returned None' if result is None else f'UNSAFE: returned {result}'}")

    passed = result is None
    print("PASS" if passed else "FAIL")
    return passed


def main():
    sweep_ok = run_sweep()
    print()
    clip_ok = check_clipped_ball_exclusion()
    print()
    noise_ok = check_noise_sweep()
    print()
    smear_ok = check_motion_smear()
    print()
    falloff_ok = check_ir_falloff()
    print()
    combined_ok = check_falloff_with_background_noise()
    print()
    distractors_ok = check_distractors()
    print()
    distractors_combined_ok = check_distractors_combined()
    print()
    online_distractor_ok = check_online_distractor_rejected()
    print()
    missing_flash_ok = check_missing_flash()
    print()
    low_confidence_ok = check_missing_flash_low_confidence_with_3_balls()
    print()
    overlap_ok = check_overlapping_balls_fail_safely()
    print()
    no_ball_ok = check_no_ball_frame()
    print()
    calibration_recovery_ok = check_calibration_recovers_distortion()
    print()
    undistortion_ok = check_undistortion_reduces_speed_error()
    print()
    per_blob_scale_ok = check_per_blob_scale_depth_varying()

    if not (sweep_ok and clip_ok and noise_ok and smear_ok and falloff_ok and combined_ok
            and distractors_ok and distractors_combined_ok and online_distractor_ok
            and missing_flash_ok and low_confidence_ok and overlap_ok and no_ball_ok
            and calibration_recovery_ok and undistortion_ok and per_blob_scale_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
