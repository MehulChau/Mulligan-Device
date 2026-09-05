"""Measure ball speed and launch angle from a strobe photograph.

Pipeline: threshold -> find round blobs -> subpixel centroid of each ->
fit a line through the centres -> derive mm/px from each blob's extent
*perpendicular* to that line (immune to motion smear along it) -> speed
from mean spacing, angle from the line's slope.
"""

import math

import cv2
import numpy as np

BALL_DIAMETER_MM = 42.67
MPS_TO_MPH = 1.0 / 0.44704

MIN_BALL_IMAGES = 3


def _measure_blob(image, contour, img_w, img_h):
    """Compute centroid, background-subtracted coverage moments, and a
    border-touching flag for one candidate contour. Returns None if the
    blob fails a basic sanity check (no positive signal)."""
    x, y, w, h = cv2.boundingRect(contour)
    touches_border = x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h

    pad = 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img_w), min(y + h + pad, img_h)
    roi = image[y0:y1, x0:x1].astype(np.float64)

    # Estimate the local background from the ROI's border pixels and
    # subtract it before computing area, moments, or centroid. A non-zero
    # or noisy background would otherwise inflate the weighted area and
    # pull the intensity-weighted centroid toward it.
    border_pixels = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    background = np.median(border_pixels)
    roi = np.clip(roi - background, 0.0, None)

    # Intensity-weighted centroid (image moments) for subpixel accuracy.
    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    total = roi.sum()
    if total <= 0:
        return None
    cx = (xs * roi).sum() / total + x0
    cy = (ys * roi).sum() / total + y0

    # Robust plateau estimate instead of a raw max: with ~7,600 saturated
    # interior pixels, a single noisy outlier is not representative -- the
    # expected maximum of that many iid noise samples sits ~4.2 sigma above
    # the true plateau (order statistic of max-of-N), which alone biases
    # area double digits once sigma reaches a handful of DN. Taking the
    # median of pixels at or above half the observed max instead is robust
    # to that one outlier and stays accurate at realistic noise levels.
    raw_max = roi.max()
    if raw_max <= 0:
        return None
    plateau = roi[roi >= 0.5 * raw_max]
    peak = np.median(plateau)
    if peak <= 0:
        return None

    # Coverage-weighted area from the raw (unthresholded, background-
    # subtracted) grayscale, not the binary mask: an anti-aliased edge
    # carries fractional pixel intensity right at the boundary, and a hard
    # binary threshold would cut through that ramp at an essentially
    # arbitrary point. Normalizing by the ROI's own plateau (not a
    # hardcoded 255) means a saturating ball -- unaffected by ambient
    # background or noise -- is read as fully covered regardless of what
    # background/noise happen to be present.
    coverage = np.clip(roi / peak, 0.0, 1.0)
    area = coverage.sum()

    # Second moments of the coverage mass about its own centroid, in raw
    # image (x, y) coordinates. Kept as a 2x2 covariance so that, once a
    # flight-line direction is known, the variance along *any* axis can be
    # recovered by projection (n^T C n) without revisiting the pixels.
    dx = xs.astype(np.float64) + x0 - cx
    dy = ys.astype(np.float64) + y0 - cy
    mxx = (coverage * dx * dx).sum() / area
    myy = (coverage * dy * dy).sum() / area
    mxy = (coverage * dx * dy).sum() / area

    return {
        "center": (cx, cy),
        "area": area,
        "cov": (mxx, myy, mxy),
        "touches_border": touches_border,
    }


def _find_blobs(image, thresholds=(40,)):
    """Find round, ball-sized blobs, trying each threshold in `thresholds`
    (highest first) and only adding blobs not already found at a higher
    one. A single global threshold can't see a bright and a faint ball at
    once (lighting falloff); trying several catches both without double-
    counting the same blob.

    Returns (blobs, excluded_border) where each blob is a dict with
    `center`, `area`, and `cov` (the 2x2 covariance of its coverage mass
    about its own centroid). Blobs touching the frame border are dropped
    entirely -- clipped by the edge, their diameter and centroid are both
    unreliable -- and counted in `excluded_border`.
    """
    img_h, img_w = image.shape
    blobs = []
    excluded_border = 0
    claimed = np.zeros(image.shape, dtype=bool)

    for threshold in sorted(set(thresholds), reverse=True):
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 20:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area_px / (perimeter * perimeter)
            if circularity < 0.7:
                continue  # reject non-round blobs (streaks, reflections, edges)

            x, y, w, h = cv2.boundingRect(contour)
            if claimed[y:y + h, x:x + w].any():
                continue  # already found at a higher threshold

            blob = _measure_blob(image, contour, img_w, img_h)
            if blob is None:
                continue
            claimed[y:y + h, x:x + w] = True

            if blob.pop("touches_border"):
                excluded_border += 1
                continue

            blobs.append(blob)

    return blobs, excluded_border


def measure(image, strobe_interval_ms, threshold=40, adaptive=False):
    """Return dict with ball_speed_mph, launch_angle_deg, ball_centers,
    scale_mm_per_px, and fit-quality confidence signals. Returns None if
    fewer than MIN_BALL_IMAGES ball images are found.

    threshold: segmentation threshold (0-255) used to find candidate blobs.
    adaptive: if True, try a descending sequence of thresholds derived from
        `threshold` instead of just one, to catch faint (e.g. far-from-
        strobe) balls a single global threshold would miss.
    """

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresholds = (threshold, threshold / 2, threshold / 4, threshold / 8) if adaptive else (threshold,)
    blobs, excluded_border = _find_blobs(image, thresholds=thresholds)
    if len(blobs) < MIN_BALL_IMAGES:
        return None

    centers = sorted((b["center"] for b in blobs), key=lambda c: c[0])
    pts = np.array(centers)

    # Least-squares line fit through the centres. Centroids are unbiased
    # under symmetric motion smear (smear stretches a blob but doesn't
    # shift its centre of mass), so fitting the line first, before scale,
    # is safe even when smear is present.
    mean = pts.mean(axis=0)
    centered = pts - mean
    # Principal direction via SVD is well-defined even for near-vertical lines.
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction  # keep pointing in +x (direction of flight)
    perp = vt[1]

    # Derive the scale from each blob's extent *perpendicular* to the
    # flight line, not from its raw area/diameter. Motion smear stretches
    # a blob only along the direction of travel -- inflating an
    # area-based diameter estimate by ~6% at driver speed and biasing
    # speed low by a similar amount -- but leaves the perpendicular extent
    # untouched. For a uniform disk of radius R, the variance along any
    # single axis through its centre is R^2/4 (a disk's marginal
    # distribution is semicircular), so R = 2*sqrt(variance_perp);
    # projecting each blob's covariance onto `perp` gives that variance
    # without revisiting pixels.
    nx, ny = perp
    perp_diameters = []
    for b in blobs:
        mxx, myy, mxy = b["cov"]
        var_perp = nx * nx * mxx + 2 * nx * ny * mxy + ny * ny * myy
        perp_diameters.append(4.0 * math.sqrt(max(var_perp, 0.0)))
    mean_diameter_px = float(np.mean(perp_diameters))
    scale_mm_per_px = BALL_DIAMETER_MM / mean_diameter_px

    # Project points onto the fitted line to get ordered positions along it.
    projections = centered @ direction
    order = np.argsort(projections)
    projections = projections[order]
    ordered_centers = pts[order]

    spacings_px = np.diff(projections)
    mean_spacing_px = float(np.mean(spacings_px))

    # Residual perpendicular distance from the fitted line, for confidence.
    residuals = centered @ perp
    line_fit_rms_px = float(np.sqrt(np.mean(residuals ** 2)))

    spacing_consistency = (
        float(np.std(spacings_px) / mean_spacing_px) if mean_spacing_px > 0 else float("inf")
    )

    speed_mm_per_flash = mean_spacing_px * scale_mm_per_px
    speed_mps = (speed_mm_per_flash / 1000.0) / (strobe_interval_ms / 1000.0)
    ball_speed_mph = speed_mps * MPS_TO_MPH

    dx, dy = direction
    # Image y increases downward; the ball rises, so flip sign for angle-up-positive.
    launch_angle_deg = math.degrees(math.atan2(-dy, dx))

    return {
        "ball_speed_mph": ball_speed_mph,
        "launch_angle_deg": launch_angle_deg,
        "ball_centers": [tuple(c) for c in ordered_centers],
        "scale_mm_per_px": scale_mm_per_px,
        "mean_diameter_px": mean_diameter_px,
        "num_ball_images": len(blobs),
        "excluded_border_blobs": excluded_border,
        "line_fit_rms_px": line_fit_rms_px,
        "spacing_consistency": spacing_consistency,
    }


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "frame.png"
    interval_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 2.9

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {image_path}")

    result = measure(img, interval_ms)
    if result is None:
        raise SystemExit("fewer than 3 ball images detected; cannot measure")

    print(f"speed:  {result['ball_speed_mph']:.1f} mph")
    print(f"angle:  {result['launch_angle_deg']:.2f} deg")
    print(f"blobs:  {result['num_ball_images']} (excluded {result['excluded_border_blobs']} touching the frame border)")
    print(f"scale:  {result['scale_mm_per_px']:.4f} mm/px")
    print(f"fit rms: {result['line_fit_rms_px']:.3f} px, spacing cv: {result['spacing_consistency']:.4f}")
