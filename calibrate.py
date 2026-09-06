"""Recover camera matrix and lens distortion coefficients from a
directory of checkerboard photographs.

A 6mm lens on this sensor gives roughly a 55-degree field of view, wide
enough that a cheap lens shows real barrel distortion toward the edges.
Ball images span nearly the full frame width, so distortion corrupts the
spacing between them directly -- at 3% distortion near the edge, that's a
3% speed error, larger than every other error source in the system
combined (see docs/hardware-readiness.md). This script produces the
calibration detect.py needs to undo that.

Usage:
    python calibrate.py <image_dir> [--pattern-cols 9] [--pattern-rows 6]
        [--square-size-mm 25] [--out calibration.json]
        [--max-reprojection-error 0.5]

See docs/calibration.md for what to actually print and photograph.
"""

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

DEFAULT_PATTERN_SIZE = (9, 6)  # internal corners (cols, rows) -- see docs/calibration.md
DEFAULT_SQUARE_SIZE_MM = 25.0
DEFAULT_MAX_REPROJECTION_ERROR_PX = 0.5

# Fewer than this and the fit is under-constrained -- one bad or
# unrepresentative photo has an outsized effect on the result.
MIN_IMAGES_RECOMMENDED = 10
MIN_IMAGES_REQUIRED = 4

# A calibration built only from centered photos is confidently wrong
# exactly where distortion is worst: the edges. Warn if the detected
# corners, pooled across every image, never reach within this fraction of
# the frame width/height from each edge.
EDGE_COVERAGE_FRACTION = 0.15

CORNER_SUBPIX_WINDOW = (11, 11)
CORNER_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build_object_points(pattern_size, square_size_mm):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm
    return objp


def find_corners_in_images(image_paths, pattern_size):
    """Run findChessboardCorners + subpixel refinement on each image.

    Returns (image_points, image_size, used_paths, failed_paths).
    """
    image_points = []
    used_paths = []
    failed_paths = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed_paths.append((path, "could not read image"))
            continue
        size = (img.shape[1], img.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            failed_paths.append((path, f"size {size} doesn't match {image_size}"))
            continue

        found, corners = cv2.findChessboardCorners(
            img, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            failed_paths.append((path, "chessboard pattern not found"))
            continue

        corners = cv2.cornerSubPix(img, corners, CORNER_SUBPIX_WINDOW, (-1, -1), CORNER_SUBPIX_CRITERIA)
        image_points.append(corners)
        used_paths.append(path)

    return image_points, image_size, used_paths, failed_paths


def check_coverage(image_points, image_size, edge_fraction=EDGE_COVERAGE_FRACTION):
    """Warn if detected corners, pooled across every image, don't reach
    near each frame edge. Distortion is worst at the edges, so a
    calibration built from twenty photos of a board in the middle of the
    frame is confidently wrong exactly where it matters."""
    all_pts = np.concatenate([p.reshape(-1, 2) for p in image_points], axis=0)
    w, h = image_size
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    warnings = []
    if x_min > edge_fraction * w:
        warnings.append(f"no corners detected near the left edge (nearest at x={x_min:.0f}px of {w}px wide)")
    if x_max < (1 - edge_fraction) * w:
        warnings.append(f"no corners detected near the right edge (nearest at x={x_max:.0f}px of {w}px wide)")
    if y_min > edge_fraction * h:
        warnings.append(f"no corners detected near the top edge (nearest at y={y_min:.0f}px of {h}px tall)")
    if y_max < (1 - edge_fraction) * h:
        warnings.append(f"no corners detected near the bottom edge (nearest at y={y_max:.0f}px of {h}px tall)")
    return warnings


def per_image_reprojection_errors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    errors = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        diff = imgp.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2).astype(np.float64)
        errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return errors


def edge_shift_px(camera_matrix, dist_coeffs, image_size, edge_fraction=0.95):
    """How far a point near the frame edge moves when undistorted -- the
    number that says whether calibration was worth doing. Reports the
    point at `edge_fraction` of the way from centre to the right edge at
    mid-height, a representative position for where a ball image actually
    gets measured (not the extreme corner, which rarely has content)."""
    w, h = image_size
    point = np.array([[[w * edge_fraction, h / 2.0]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(point, camera_matrix, dist_coeffs, P=camera_matrix)
    dx = undistorted[0, 0, 0] - point[0, 0, 0]
    dy = undistorted[0, 0, 1] - point[0, 0, 1]
    return math.hypot(dx, dy), (point[0, 0, 0], point[0, 0, 1])


def calibrate(image_paths, pattern_size, square_size_mm, max_reprojection_error_px):
    """Run the full calibration pipeline and return the calibration dict,
    or raise SystemExit with an explanatory message if the result isn't
    trustworthy enough to use.
    """
    if len(image_paths) < MIN_IMAGES_RECOMMENDED:
        print(f"WARNING: only {len(image_paths)} image(s) given; "
              f"{MIN_IMAGES_RECOMMENDED}+ recommended for a well-constrained fit.")

    image_points, image_size, used_paths, failed_paths = find_corners_in_images(image_paths, pattern_size)
    for path, reason in failed_paths:
        print(f"WARNING: skipping {path}: {reason}")

    if len(image_points) < MIN_IMAGES_REQUIRED:
        raise SystemExit(
            f"only {len(image_points)} image(s) yielded a detected board; "
            f"need at least {MIN_IMAGES_REQUIRED} to calibrate at all, and "
            f"{MIN_IMAGES_RECOMMENDED}+ for a fit worth trusting."
        )
    if len(image_points) < MIN_IMAGES_RECOMMENDED:
        print(f"WARNING: only {len(image_points)} image(s) had a detected board "
              f"(of {len(image_paths)} given); {MIN_IMAGES_RECOMMENDED}+ recommended.")

    for warning in check_coverage(image_points, image_size):
        print(f"WARNING: {warning}")

    objp = build_object_points(pattern_size, square_size_mm)
    object_points = [objp] * len(image_points)

    # K3 fixed at 0: a reasonable, standard simplification for a lens this
    # far from fisheye (~55deg FOV) -- the 6th-order term is otherwise
    # weakly constrained and prone to trading off against K2 (verified
    # empirically: freeing it recovered a K3 an order of magnitude away
    # from the true 0 while barely changing reprojection error).
    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=cv2.CALIB_FIX_K3,
    )

    per_image_errors = per_image_reprojection_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs,
    )
    print(f"reprojection error: {reprojection_error:.4f}px overall "
          f"(per-image min={min(per_image_errors):.4f}, max={max(per_image_errors):.4f})")

    shift_px, edge_point = edge_shift_px(camera_matrix, dist_coeffs, image_size)
    print(f"distortion effect: a point at x={edge_point[0]:.0f}px "
          f"(95% of frame width) moves {shift_px:.2f}px ({shift_px / image_size[0] * 100:.2f}% "
          "of frame width) when undistorted")

    if reprojection_error > max_reprojection_error_px:
        raise SystemExit(
            f"reprojection error {reprojection_error:.4f}px exceeds the "
            f"{max_reprojection_error_px}px threshold -- refusing to write this calibration. "
            "A bad calibration is worse than none: it silently distorts every "
            "subsequent measurement instead of leaving them alone. Retake photos with "
            "sharper focus, better lighting, and a rigid (non-warped) board."
        )

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "reprojection_error_px": reprojection_error,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "num_images_used": len(image_points),
        "pattern_size": list(pattern_size),
        "square_size_mm": square_size_mm,
        "edge_shift_px": shift_px,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", help="directory of checkerboard photographs")
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_SIZE[0],
                         help=f"internal corners per row (default {DEFAULT_PATTERN_SIZE[0]})")
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_SIZE[1],
                         help=f"internal corners per column (default {DEFAULT_PATTERN_SIZE[1]})")
    parser.add_argument("--square-size-mm", type=float, default=DEFAULT_SQUARE_SIZE_MM,
                         help=f"physical checkerboard square size in mm (default {DEFAULT_SQUARE_SIZE_MM})")
    parser.add_argument("--out", default="calibration.json", help="output calibration file (default calibration.json)")
    parser.add_argument("--max-reprojection-error", type=float, default=DEFAULT_MAX_REPROJECTION_ERROR_PX,
                         help=f"refuse to write a calibration above this, in px (default {DEFAULT_MAX_REPROJECTION_ERROR_PX})")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in glob.glob(os.path.join(args.image_dir, "*"))
        if p.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_paths:
        raise SystemExit(f"no images found in {args.image_dir}")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    result = calibrate(image_paths, pattern_size, args.square_size_mm, args.max_reprojection_error)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out} ({result['num_images_used']} images used, "
          f"reprojection error {result['reprojection_error_px']:.4f}px)")


if __name__ == "__main__":
    main()
