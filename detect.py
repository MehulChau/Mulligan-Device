"""Measure ball speed and launch angle from a strobe photograph.

Pipeline: threshold -> find round blobs -> subpixel centroid of each ->
derive mm/px from the ball's known physical diameter -> fit a line through
the centres -> speed from mean spacing, angle from the line's slope.
"""

import math

import cv2
import numpy as np

BALL_DIAMETER_MM = 42.67
MPS_TO_MPH = 1.0 / 0.44704

MIN_BALL_IMAGES = 3


def _find_blobs(image, threshold=40):
    """Threshold and return (center, radius_px, area) for round, ball-sized blobs."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue  # reject non-round blobs (streaks, reflections, edges)

        x, y, w, h = cv2.boundingRect(contour)
        pad = 2
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, image.shape[1]), min(y + h + pad, image.shape[0])
        roi = image[y0:y1, x0:x1].astype(np.float64)

        # Intensity-weighted centroid (image moments) for subpixel accuracy.
        ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        total = roi.sum()
        if total <= 0:
            continue
        cx = (xs * roi).sum() / total + x0
        cy = (ys * roi).sum() / total + y0

        # Area from the raw (unthresholded) grayscale coverage, not the
        # binary mask: an anti-aliased edge carries fractional pixel
        # intensity right at the boundary, and a hard binary threshold
        # would cut through that ramp at an essentially arbitrary point,
        # biasing the recovered radius by a threshold-dependent amount.
        # Summing raw intensity (0-255) and dividing by 255 recovers the
        # true covered area regardless of the threshold used to segment.
        weighted_area = roi.sum() / 255.0
        radius_px = math.sqrt(weighted_area / math.pi)
        blobs.append({"center": (cx, cy), "radius_px": radius_px, "area": area})

    return blobs


def measure(image, strobe_interval_ms, threshold=40):
    """Return dict with ball_speed_mph, launch_angle_deg, ball_centers,
    scale_mm_per_px, and fit-quality confidence signals. Returns None if
    fewer than MIN_BALL_IMAGES ball images are found."""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blobs = _find_blobs(image, threshold=threshold)
    if len(blobs) < MIN_BALL_IMAGES:
        return None

    # Self-calibrate scale from the ball's known real diameter, averaged
    # across all detected ball images. This makes the measurement
    # independent of the assumed camera-to-ball distance.
    mean_diameter_px = 2.0 * np.mean([b["radius_px"] for b in blobs])
    scale_mm_per_px = BALL_DIAMETER_MM / mean_diameter_px

    centers = sorted((b["center"] for b in blobs), key=lambda c: c[0])
    pts = np.array(centers)

    # Least-squares line fit through the centres.
    mean = pts.mean(axis=0)
    centered = pts - mean
    # Principal direction via SVD is well-defined even for near-vertical lines.
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction  # keep pointing in +x (direction of flight)

    # Project points onto the fitted line to get ordered positions along it.
    projections = centered @ direction
    order = np.argsort(projections)
    projections = projections[order]
    ordered_centers = pts[order]

    spacings_px = np.diff(projections)
    mean_spacing_px = float(np.mean(spacings_px))

    # Residual perpendicular distance from the fitted line, for confidence.
    perp = vt[1]
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
    print(f"blobs:  {result['num_ball_images']}")
    print(f"scale:  {result['scale_mm_per_px']:.4f} mm/px")
    print(f"fit rms: {result['line_fit_rms_px']:.3f} px, spacing cv: {result['spacing_consistency']:.4f}")
