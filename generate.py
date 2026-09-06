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

# Effective focal length in pixels, from the same pinhole relation used
# for SCALE_MM_PER_PX above (f_px = f_mm * sensor_width_px / sensor_width_mm).
# generate.py itself always draws an ideal, distortion-free pinhole image;
# this and DEFAULT_CAMERA_MATRIX exist only to define *where* a lens's
# barrel distortion would move a point, for the optional `distortion`
# parameter on generate_frame() and for generate_checkerboard_frame().
FOCAL_LENGTH_PX = LENS_FOCAL_LENGTH_MM / SENSOR_WIDTH_MM * SENSOR_WIDTH_PX
DEFAULT_CAMERA_MATRIX = [
    [FOCAL_LENGTH_PX, 0.0, SENSOR_WIDTH_PX / 2.0],
    [0.0, FOCAL_LENGTH_PX, SENSOR_HEIGHT_PX / 2.0],
    [0.0, 0.0, 1.0],
]


def _distort_image(image, camera_matrix, dist_coeffs):
    """Warp a clean, undistorted synthetic image into what a lens with
    `dist_coeffs` would actually have captured of the same scene.

    There's no direct "distort an image" function in OpenCV, only the
    inverse (cv2.undistort corrects a real photo). This builds it from
    cv2.undistortPoints instead: for every pixel in the *output*
    (distorted) image, find where in the *clean* (undistorted) image it
    must have come from, then resample there via cv2.remap. That's the
    exact geometric inverse of cv2.undistort, so undistorting the result
    afterward with the same camera_matrix/dist_coeffs recovers the
    original up to interpolation error.

    An earlier version of this distorted each ball's *centre* individually
    and drew an ordinary undistorted disk there, which is simpler but
    wrong in a way that matters here: it leaves every feature's *shape*
    undistorted, which a real lens never would, and cv2.undistort then
    can't cleanly invert a per-feature shape distortion that was never
    actually applied -- verified this empirically: undistorting made the
    measured speed error *worse*, not better. Warping the whole rendered
    image once, after every ball is drawn, avoids that mismatch entirely.
    """
    h, w = image.shape
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).reshape(-1, 1, 2)
    source_coords = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    map_x = source_coords[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y = source_coords[:, 0, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


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


def _draw_laser_dot(image, cx, cy, radius=4.0, value=255):
    """A small, bright, round dot marking where the ball was addressed.
    Reuses the same anti-aliased disk drawer as a real ball image -- it's
    round and bright for the same optical reasons, just much smaller."""
    _draw_ball(image, cx, cy, radius, value)


def _draw_mat_reflection(image, cx, cy, sigma_x=45.0, sigma_y=30.0, peak=70.0):
    """A dim, diffuse smudge from light reflecting off the hitting mat --
    unlike a ball image, it has no hard edge at all, just a soft Gaussian
    falloff, and is dim enough to sit well below a ball's saturating peak.
    """
    h, w = image.shape
    margin = int(4 * max(sigma_x, sigma_y))
    x0, x1 = max(int(cx - margin), 0), min(int(cx + margin), w)
    y0, y1 = max(int(cy - margin), 0), min(int(cy + margin), h)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xs - cx) ** 2) / (2 * sigma_x ** 2) + ((ys - cy) ** 2) / (2 * sigma_y ** 2)))
    patch = (g * peak).astype(np.uint8)
    image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], patch)


def _draw_clubhead_edge(image, value=230):
    """A bright, hard-edged wedge intruding from a frame corner --
    part of the clubhead clipping the frame during the swing. Triangular
    and edge-touching by construction, unlike a ball: it should fail
    circularity even before the border-touching check would exclude it."""
    h, w = image.shape
    apex = (w - 1, 0)
    base_half = 90
    depth = 130
    pts = np.array([
        [apex[0], apex[1]],
        [apex[0] - base_half, apex[1]],
        [apex[0], apex[1] + depth],
    ], dtype=np.int32)
    cv2.fillPoly(image, [pts], value)


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
    laser_dot=False,
    laser_dot_radius_px=4.0,
    laser_dot_pos_px=None,
    mat_reflection=False,
    mat_reflection_pos_px=None,
    clubhead_edge=False,
    missing_flash_index=None,
    dist_coeffs=None,
    camera_matrix=None,
    depth_profile=None,
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
    laser_dot / mat_reflection / clubhead_edge: optional distractors seen
        on a real range frame -- a bright placement-marker dot, a dim
        diffuse mat reflection, and a bright edge-touching clubhead
        intrusion, respectively.
    missing_flash_index: if set, that flash index (0-based) doesn't fire --
        nothing is drawn for it, but unlike an off-frame flash this is a
        mid-sequence gap the ground truth still records, for testing how
        the detector handles an uneven spacing.
    dist_coeffs / camera_matrix: optional (k1, k2, p1, p2, k3) lens
        distortion applied to each ball's ideal pinhole position before
        drawing (camera_matrix defaults to DEFAULT_CAMERA_MATRIX). None
        (default) draws the same distortion-free pinhole image as always.
    depth_profile: optional list of `num_flashes` per-flash relative
        distance factors (1.0 = the nominal CAMERA_DISTANCE_M), for
        testing a flight where the ball moves toward or away from the
        camera. Each flash's apparent diameter is scaled by 1/factor, and
        the pixel gap between consecutive flashes uses the *average* of
        the two endpoints' scales -- the same rule detect.py's per-blob
        scale derivation uses -- so the flight is self-consistent at the
        constant real-world speed the caller asked for. None (default,
        equivalent to all 1.0) reproduces the original constant-depth
        behaviour exactly.

    Returns (image, ground_truth_dict).
    """
    image = np.zeros((height, width), dtype=np.uint8)
    if camera_matrix is None:
        camera_matrix = DEFAULT_CAMERA_MATRIX

    angle_rad = math.radians(launch_angle_deg)
    direction = (math.cos(angle_rad), -math.sin(angle_rad))  # y flips: image y increases downward

    speed_mps = ball_speed_mph * MPH_TO_MPS
    speed_px_per_s = (speed_mps * 1000.0) / SCALE_MM_PER_PX  # used by motion smear only; see note there
    mm_per_flash = speed_mps * 1000.0 * (strobe_interval_ms / 1000.0)

    if depth_profile is None:
        depth_profile = [1.0] * num_flashes
    scales = [SCALE_MM_PER_PX * f for f in depth_profile]
    radii_px = [(BALL_DIAMETER_MM / 2.0) / s for s in scales]

    if start_pos_px is None:
        # Half a ball diameter from the edge, plus a small border-safety
        # buffer (see BORDER_SAFETY_PX above). num_flashes=0 has no first
        # radius to key off of, so falls back to the nominal ball size --
        # the margin is moot either way with nothing to draw.
        first_radius = radii_px[0] if radii_px else BALL_DIAMETER_PX / 2.0
        margin = first_radius + BORDER_SAFETY_PX
        start_x = margin
        start_y = height - margin
        start_pos_px = (start_x, start_y)

    # Centres in the ideal (undistorted, pinhole) image: constant
    # real-world mm per flash, converted to pixels via each gap's local
    # (endpoint-averaged) scale, so the flight is self-consistent under
    # depth_profile. Everything is drawn at these ideal positions; lens
    # distortion (if requested) is applied once, as a whole-image warp,
    # after all drawing is done -- see _distort_image for why that's
    # correct where distorting each centre individually isn't.
    centers = [start_pos_px]
    for i in range(1, num_flashes):
        local_scale = (scales[i - 1] + scales[i]) / 2.0
        step_px = mm_per_flash / local_scale
        prev = centers[-1]
        centers.append((prev[0] + step_px * direction[0], prev[1] + step_px * direction[1]))

    statuses = [_ball_visibility(cx, cy, r, width, height) for (cx, cy), r in zip(centers, radii_px)]

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

    # Motion smear uses the nominal (depth_profile-independent) scale --
    # combining a changing-depth flight with motion smear isn't something
    # any test exercises, so the small resulting inconsistency (smear
    # distance not itself depth-scaled) is an accepted simplification
    # rather than something worth the extra bookkeeping.
    smear_px = speed_px_per_s * (flash_duration_us * 1e-6) if flash_duration_us > 0 else 0.0

    balls = []
    for i, ((cx, cy), status, intensity, radius_px) in enumerate(zip(centers, statuses, intensities, radii_px)):
        if i == missing_flash_index:
            # The strobe didn't fire for this exposure: nothing is drawn,
            # but this is a mid-sequence gap, not an off-frame flash --
            # record it distinctly so ground truth still reflects reality.
            balls.append({"center_px": (cx, cy), "status": "misfired", "intensity": 0.0})
            continue
        balls.append({
            "center_px": (cx, cy), "status": status, "intensity": intensity,
            "diameter_px": radius_px * 2.0,
        })
        if status == "off_frame":
            continue  # nothing to draw -- it isn't in the image at all
        value = int(round(min(max(intensity, 0.0), 255.0)))
        if smear_px > 0:
            _draw_ball_smeared(image, cx, cy, radius_px, direction, smear_px, value=value)
        else:
            _draw_ball(image, cx, cy, radius_px, value=value)

    distractors = []
    if laser_dot:
        # Offset up and to the side from the first ball's position, since
        # start_pos_px sits near the bottom edge -- an offset downward
        # would land outside the frame.
        pos = laser_dot_pos_px or (start_pos_px[0] + 60.0, start_pos_px[1] - 90.0)
        _draw_laser_dot(image, pos[0], pos[1], radius=laser_dot_radius_px)
        distractors.append({"type": "laser_dot", "center_px": pos, "radius_px": laser_dot_radius_px})
    if mat_reflection:
        pos = mat_reflection_pos_px or (start_pos_px[0] + 250.0, start_pos_px[1] - 90.0)
        _draw_mat_reflection(image, pos[0], pos[1])
        distractors.append({"type": "mat_reflection", "center_px": pos})
    if clubhead_edge:
        _draw_clubhead_edge(image)
        distractors.append({"type": "clubhead_edge"})

    if dist_coeffs is not None:
        image = _distort_image(image, camera_matrix, dist_coeffs)

    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0.0, 255.0).astype(np.uint8)

    ground_truth = {
        "ball_speed_mph": ball_speed_mph,
        "launch_angle_deg": launch_angle_deg,
        "num_flashes": num_flashes,
        "strobe_interval_ms": strobe_interval_ms,
        "balls": balls,
        "ball_diameter_px": BALL_DIAMETER_PX,
        "scale_mm_per_px": SCALE_MM_PER_PX,
        "image_width": width,
        "image_height": height,
        "noise_sigma": noise_sigma,
        "flash_duration_us": flash_duration_us,
        "smear_px": smear_px,
        "dist_coeffs": list(dist_coeffs) if dist_coeffs is not None else None,
        "camera_matrix": camera_matrix if dist_coeffs is not None else None,
        "depth_profile": depth_profile,
        "ir_falloff": ir_falloff,
        "strobe_source_px": strobe_source_px,
        "distractors": distractors,
        "missing_flash_index": missing_flash_index,
    }
    return image, ground_truth


_POLY_RENDER_SHIFT = 4  # sub-pixel fixed-point bits for cv2.fillPoly, see _fill_poly_subpixel


def _fill_poly_subpixel(image, quad_px, color):
    """cv2.fillPoly requires integer points; naively rounding each corner
    to the nearest pixel before filling adds up to ~0.5px of noise per
    corner, which is exactly the kind of error a calibration recovery
    test can't afford (early testing showed it measurably corrupting the
    recovered distortion coefficients). cv2.fillPoly's `shift` parameter
    takes coordinates pre-multiplied by 2**shift and rounds once at that
    finer resolution instead, at effectively full float precision for a
    16x (shift=4) finer grid.
    """
    factor = 1 << _POLY_RENDER_SHIFT
    pts = np.round(np.asarray([quad_px], dtype=np.float64) * factor).astype(np.int32)
    cv2.fillPoly(image, pts, color, lineType=cv2.LINE_AA, shift=_POLY_RENDER_SHIFT)


def generate_checkerboard_frame(
    pattern_size=(9, 6),
    square_size_mm=25.0,
    board_distance_m=CAMERA_DISTANCE_M,
    board_tilt_deg=(0.0, 0.0),
    board_offset_mm=(0.0, 0.0),
    dist_coeffs=None,
    camera_matrix=None,
    width=SENSOR_WIDTH_PX,
    height=SENSOR_HEIGHT_PX,
):
    """Draw one synthetic photo of a checkerboard held at a given pose in
    front of the camera, optionally through the same lens distortion
    generate_frame() uses, so calibrate.py can be tested with no camera.

    `pattern_size` is OpenCV's own convention -- the number of *internal*
    corners (cols, rows), where 4 squares meet -- so it can be passed
    straight through to cv2.findChessboardCorners. A board with that many
    internal corners has (cols+1, rows+1) total squares.

    A real calibration needs the board photographed at *varied angles*,
    not just varied positions: a fronto-parallel board photographed only
    at different image positions/scales under-constrains the fit (focal
    length and distortion trade off against each other; verified this
    empirically -- it recovers a self-consistent but badly wrong camera
    matrix and distortion with no warning). `board_tilt_deg` (rotation
    about the board's own x and y axes, i.e. tilting it away from the
    camera) is what breaks that degeneracy, exactly as it does for a
    person moving a real board in front of a real camera.

    The board is modeled as a real, physical, planar object at
    `board_distance_m` along the camera axis (plus `board_offset_mm`
    lateral shift and `board_tilt_deg` rotation) and projected with
    cv2.projectPoints -- the same operation, run forward, that
    cv2.calibrateCamera solves the inverse of -- rather than warping a
    flat image, so perspective and lens distortion combine correctly.

    Returns (image, ground_truth_dict) with the object-space (mm) and
    true image-space (px, undistorted and distorted) position of every
    internal corner, plus the pose and camera model used to produce them.
    """
    if camera_matrix is None:
        camera_matrix = DEFAULT_CAMERA_MATRIX
    camera_matrix_np = np.array(camera_matrix, dtype=np.float64)
    dist_np = np.array(dist_coeffs, dtype=np.float64) if dist_coeffs is not None else np.zeros(5)

    squares_x, squares_y = pattern_size[0] + 1, pattern_size[1] + 1
    xs_mm = (np.arange(squares_x + 1) - squares_x / 2.0) * square_size_mm
    ys_mm = (np.arange(squares_y + 1) - squares_y / 2.0) * square_size_mm
    grid_mm = np.array([[x, y, 0.0] for y in ys_mm for x in xs_mm], dtype=np.float64)

    tilt_x_rad, tilt_y_rad = math.radians(board_tilt_deg[0]), math.radians(board_tilt_deg[1])
    rx = np.array([[1, 0, 0], [0, math.cos(tilt_x_rad), -math.sin(tilt_x_rad)],
                   [0, math.sin(tilt_x_rad), math.cos(tilt_x_rad)]])
    ry = np.array([[math.cos(tilt_y_rad), 0, math.sin(tilt_y_rad)], [0, 1, 0],
                   [-math.sin(tilt_y_rad), 0, math.cos(tilt_y_rad)]])
    rvec, _ = cv2.Rodrigues(ry @ rx)
    tvec = np.array([board_offset_mm[0], board_offset_mm[1], board_distance_m * 1000.0])

    grid_distorted, _ = cv2.projectPoints(grid_mm, rvec, tvec, camera_matrix_np, dist_np)
    grid_distorted = grid_distorted.reshape(len(ys_mm), len(xs_mm), 2)
    grid_undistorted, _ = cv2.projectPoints(grid_mm, rvec, tvec, camera_matrix_np, np.zeros(5))
    grid_undistorted = grid_undistorted.reshape(len(ys_mm), len(xs_mm), 2)

    image = np.full((height, width), 255, dtype=np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                continue  # alternate squares: leave the white background showing
            quad = [grid_distorted[row, col], grid_distorted[row, col + 1],
                    grid_distorted[row + 1, col + 1], grid_distorted[row + 1, col]]
            _fill_poly_subpixel(image, quad, 0)

    corners_mm, corners_undistorted_px, corners_distorted_px = [], [], []
    for row in range(1, squares_y):
        for col in range(1, squares_x):
            corners_mm.append((float(xs_mm[col]), float(ys_mm[row])))
            corners_undistorted_px.append(tuple(grid_undistorted[row, col]))
            corners_distorted_px.append(tuple(grid_distorted[row, col]))

    ground_truth = {
        "pattern_size": list(pattern_size),
        "square_size_mm": square_size_mm,
        "board_distance_m": board_distance_m,
        "board_tilt_deg": list(board_tilt_deg),
        "board_offset_mm": list(board_offset_mm),
        "camera_matrix": camera_matrix,
        "dist_coeffs": list(dist_coeffs) if dist_coeffs is not None else None,
        "corners_object_mm": corners_mm,
        "corners_undistorted_px": corners_undistorted_px,
        "corners_distorted_px": corners_distorted_px,
        "image_width": width,
        "image_height": height,
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
