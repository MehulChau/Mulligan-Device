"""Measure ball speed and launch angle from a strobe photograph.

Pipeline: threshold -> find round blobs -> subpixel centroid of each ->
reject blobs whose size is inconsistent with the rest (relative to the
modal detected size, not a fixed expected diameter) -> robustly fit a
line through the surviving centres, keeping only the largest mutually-
collinear/evenly-spaced consensus set -> derive mm/px from each consensus
blob's extent *perpendicular* to that line (immune to motion smear along
it) -> speed from spacing, angle from the line's slope.
"""

import math

import cv2
import numpy as np

BALL_DIAMETER_MM = 42.67
MPS_TO_MPH = 1.0 / 0.44704

MIN_BALL_IMAGES = 3

# Spacing coefficient of variation above this indicates a missing/misfired
# exposure, not just measurement noise: a clean flash sequence's spacing
# consistency is normally well under 0.05 even with realistic noise.
UNEVEN_SPACING_THRESHOLD = 0.15

# A blob's size must fall within this band of the *modal* detected size to
# survive -- relative, not absolute, since the modal size is whatever the
# real ball happens to measure at this camera's actual (unknown) distance.
# A hardcoded expected diameter would quietly reintroduce the fixed-
# distance assumption the perpendicular-extent scale derivation exists to
# remove. The band is wide (0.6x-1.6x) because it only needs to separate
# one real, consistent physical object from something else entirely (a
# laser dot at ~8px next to balls at ~99px is off by 12x), not to pin down
# an exact size.
SIZE_CONSISTENCY_BAND = (0.6, 1.6)

# Perpendicular distance (px) from a candidate line within which a blob
# counts as an inlier during the RANSAC line fit.
RANSAC_INLIER_THRESHOLD_PX = 5.0


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
    # subtract it before computing area, moments, or centroid. Keep the
    # residual *signed* here -- clipping negative excursions to zero would
    # rectify the noise: every background pixel that happens to fluctuate
    # above the background estimate then contributes a small positive
    # weight at a large lever arm (d^2 in the moments) with no
    # corresponding negative contribution to cancel it, systematically
    # inflating area/variance and reading speed low. Signed noise around a
    # correctly-estimated background averages to ~0 in these sums instead.
    border_pixels = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    background = np.median(border_pixels)
    roi = roi - background

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
    # background/noise happen to be present. Only the upper bound is
    # clipped (a pixel can't be more than fully covered) -- clipping the
    # lower bound at 0 would rectify noise the same way the background
    # subtraction above avoids, and this coverage sum feeds the same
    # second moments.
    coverage = np.clip(roi / peak, None, 1.0)
    area = coverage.sum()

    # Second moments of the coverage mass about its own centroid, in raw
    # image (x, y) coordinates. Kept as a 2x2 covariance so that, once a
    # flight-line direction is known, the variance along *any* axis can be
    # recovered by projection (n^T C n) without revisiting the pixels.
    #
    # Bound the lever arm before summing: a d^2 weight means even a small
    # residual far from the centroid (e.g. a bit of the ROI's noisy
    # background pulled in when noise widens the source contour's bounding
    # box) contributes disproportionately. Restricting to pixels within
    # ~1.3x the blob's own rough radius keeps that lever arm bounded; the
    # signed residual above is what stops it from being one-sided, this
    # just keeps stray far-field noise from reaching the sum at all.
    dx = xs.astype(np.float64) + x0 - cx
    dy = ys.astype(np.float64) + y0 - cy
    rough_radius = math.sqrt(max(area, 0.0) / math.pi)
    window = (dx * dx + dy * dy) <= (1.3 * rough_radius) ** 2
    coverage_windowed = np.where(window, coverage, 0.0)
    area = coverage_windowed.sum()
    if area <= 0:
        return None
    mxx = (coverage_windowed * dx * dx).sum() / area
    myy = (coverage_windowed * dy * dy).sum() / area
    mxy = (coverage_windowed * dx * dy).sum() / area

    return {
        "center": (cx, cy),
        "area": area,
        "cov": (mxx, myy, mxy),
        "touches_border": touches_border,
    }


def _adaptive_thresholds(image, min_threshold):
    """Derive a descending threshold sequence from the frame's own
    statistics instead of blind halving. Estimate background level and
    noise sigma via median and MAD (a scaled median absolute deviation is
    a standard robust sigma estimator, insensitive to the bright ball
    pixels that are themselves outliers relative to the background). A
    fixed halving sequence like (40, 20, 10, 5) can step below the actual
    background level -- with an ambient background of 25, for instance,
    every threshold under 25 sees the *entire* frame as foreground,
    producing one giant contour that can even pass the circularity filter
    (a near-full-frame region isn't far from that 0.7 cutoff). Stepping
    down in multiples of sigma above the *measured* background, floored at
    a healthy margin above it, can't repeat that failure.
    """
    background = float(np.median(image))
    mad = float(np.median(np.abs(image.astype(np.float64) - background)))
    sigma = max(1.4826 * mad, 1.0)  # MAD->sigma factor for Gaussian noise; floor for a flat/clean frame

    floor = background + 3.0 * sigma  # never step down into the noise floor itself
    candidates = (background + k * sigma for k in (10.0, 6.0, 4.0, 3.0))
    thresholds = {min(255.0, max(t, floor)) for t in candidates}
    # Include the caller's own threshold too, since it's usually tuned to
    # the brightest expected signal (e.g. a saturating ball) -- but only if
    # it clears the same floor; otherwise it's exactly the failure mode
    # above (a fixed threshold sitting below the measured background).
    if min_threshold >= floor:
        thresholds.add(float(min_threshold))
    return sorted(thresholds, reverse=True)


def _isotropic_diameter(blob):
    """A direction-agnostic size estimate for a blob, used only to compare
    blobs' sizes to each other (not to derive scale -- that still happens
    via the perpendicular-projected variance once a flight line is known).
    For a uniform disk, the average of the two axis variances equals the
    same R^2/4 as either axis alone; for a smeared blob it's inflated by
    roughly the same amount regardless of orientation, so real ball images
    under the same smear stay mutually consistent by this measure even
    though it isn't the precise perpendicular value used for scale."""
    mxx, myy, _ = blob["cov"]
    return 4.0 * math.sqrt(max((mxx + myy) / 2.0, 0.0))


def _filter_by_size_consistency(blobs, band=SIZE_CONSISTENCY_BAND):
    """Reject blobs whose size is far from the modal detected size.

    All genuine ball images in one frame are the same physical object --
    IR falloff changes their brightness, not their diameter -- so they
    cluster tightly in size regardless of what that size actually is at
    this camera's (unknown) distance. The median of the detected sizes is
    a stand-in for the mode; establishing it needs at least MIN_BALL_IMAGES
    blobs, which the caller already guarantees before calling this.
    """
    if len(blobs) < MIN_BALL_IMAGES:
        return blobs, 0
    sizes = np.array([_isotropic_diameter(b) for b in blobs])
    modal_size = float(np.median(sizes))
    lo, hi = band[0] * modal_size, band[1] * modal_size
    keep = (sizes >= lo) & (sizes <= hi)
    kept_blobs = [b for b, k in zip(blobs, keep) if k]
    return kept_blobs, len(blobs) - len(kept_blobs)


def _fit_line_ransac(pts, inlier_threshold_px=RANSAC_INLIER_THRESHOLD_PX):
    """Find the largest set of mutually-collinear points among `pts` by
    exhaustively trying every pair as a line hypothesis and scoring it by
    how many points fall within `inlier_threshold_px` of it.

    Real ball images are both collinear and evenly spaced -- two strong
    constraints a stray blob (a distractor that slipped past the size
    filter, say) almost never satisfies by accident. Detected blob counts
    here are small (a handful, rarely more than a dozen), so exhaustively
    trying every pair is cheap and more reliable than random sampling --
    it's not really "RANSAC" so much as its small-n limit, but the idea
    (hypothesize from a minimal sample, score by consensus) is the same.

    Returns a boolean inlier mask for the best line found.
    """
    n = len(pts)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            norm = math.hypot(d[0], d[1])
            if norm < 1e-9:
                continue  # coincident points can't define a direction
            d = d / norm
            perp_candidate = np.array([-d[1], d[0]])
            dist = np.abs((pts - pts[i]) @ perp_candidate)
            inliers = dist <= inlier_threshold_px
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
    return best_inliers


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

        max_area_px = 0.05 * image.shape[0] * image.shape[1]
        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 20 or area_px > max_area_px:
                # Too big to be a ball at any plausible scale -- a
                # threshold that dips near or below the background level
                # can turn the *entire* frame into one contour, and that
                # contour's circularity can be high enough to pass the
                # roundness filter below (a near-full-frame rectangle
                # isn't far from the 0.7 cutoff), so area is checked first.
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
    fewer than MIN_BALL_IMAGES ball images survive detection, size
    filtering, or the line-fit consensus requirement.

    threshold: segmentation threshold (0-255) used to find candidate blobs.
    adaptive: if True, try a descending sequence of thresholds derived from
        the frame's own background/noise statistics instead of just one,
        to catch faint (e.g. far-from-strobe) balls a single global
        threshold would miss.

    The result's excluded_border_blobs, excluded_size_blobs,
    excluded_outlier_blobs, and consensus_size together say *where* blobs
    were lost -- border clipping, size inconsistency (e.g. a distractor),
    or failing the line-fit consensus -- which on real hardware is what
    distinguishes a bad reading caused by detection from one caused by
    measurement.
    """

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresholds = _adaptive_thresholds(image, threshold) if adaptive else (threshold,)
    blobs, excluded_border = _find_blobs(image, thresholds=thresholds)
    if len(blobs) < MIN_BALL_IMAGES:
        return None

    # Relative size filter: reject blobs far from the modal detected size
    # before they ever reach the line fit. A laser dot or other small
    # bright distractor is wrong by a large factor (often 10x+) regardless
    # of what the real ball's size turns out to be at this camera's
    # (unknown) distance -- deliberately not compared to any fixed
    # expected diameter, since that would reintroduce the fixed-distance
    # assumption the perpendicular-extent scale derivation removes.
    blobs, excluded_size = _filter_by_size_consistency(blobs)
    if len(blobs) < MIN_BALL_IMAGES:
        return None

    pts = np.array([b["center"] for b in blobs])

    # Robust line fit: real ball images are both collinear and evenly
    # spaced, two constraints a stray blob (one that happened to survive
    # size filtering) almost never satisfies at once. Fitting through
    # every detected blob would let a single survivor corrupt the
    # direction, the perpendicular scale, and the spacing all at once, so
    # find the largest mutually-collinear subset first and fit only on
    # that -- refitting via SVD for the final, precise line.
    inlier_mask = _fit_line_ransac(pts)
    num_candidates = len(blobs)
    num_inliers = int(inlier_mask.sum())
    excluded_outliers = num_candidates - num_inliers

    # An absolute floor and a majority requirement: a consensus set below
    # MIN_BALL_IMAGES can't define a trustworthy line at all, and a
    # "consensus" that isn't even a majority of what was detected means
    # the outliers outnumber the real balls -- not a shot to guess at.
    if num_inliers < MIN_BALL_IMAGES or 2 * num_inliers <= num_candidates:
        return None

    blobs = [b for b, keep in zip(blobs, inlier_mask) if keep]
    pts = pts[inlier_mask]

    # Least-squares line fit through the consensus centres. Centroids are
    # unbiased under symmetric motion smear (smear stretches a blob but
    # doesn't shift its centre of mass), so fitting the line first, before
    # scale, is safe even when smear is present.
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

    # A uniform flash sequence gives near-identical spacing between
    # consecutive balls (spacing_consistency close to 0 in practice). One
    # missing/misfired exposure leaves a single gap at roughly double
    # width, which the *mean* spacing would blend into every interval --
    # for a 5-flash shot missing one exposure, that's a +33% speed error,
    # not measurement noise. When spacing is this uneven, the *minimum*
    # gap is the right estimate of one genuine flash interval: any gap
    # spanning a missed exposure can only be a multiple of it, never
    # smaller, so the smallest observed gap is never itself corrupted by
    # a skip (as long as at least one true single-interval gap survived).
    #
    # With only 3 balls (2 gaps), though, spacing_consistency is computed
    # from a single pair of gaps -- a noisy statistic on its own -- and
    # the minimum of just 2 values is itself biased low. Below 4 balls
    # (3+ gaps), don't apply the correction; flag the frame as
    # low-confidence instead so the caller knows not to trust it blindly.
    uneven_spacing = spacing_consistency > UNEVEN_SPACING_THRESHOLD
    missing_flash_detected = uneven_spacing and len(blobs) >= 4
    low_confidence_spacing = uneven_spacing and len(blobs) < 4
    representative_spacing_px = float(np.min(spacings_px)) if missing_flash_detected else mean_spacing_px

    speed_mm_per_flash = representative_spacing_px * scale_mm_per_px
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
        "excluded_size_blobs": excluded_size,
        "excluded_outlier_blobs": excluded_outliers,
        "consensus_size": num_inliers,
        "line_fit_rms_px": line_fit_rms_px,
        "spacing_consistency": spacing_consistency,
        "missing_flash_detected": missing_flash_detected,
        "low_confidence_spacing": low_confidence_spacing,
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
        raise SystemExit("fewer than 3 ball images survived detection/filtering; cannot measure")

    print(f"speed:  {result['ball_speed_mph']:.1f} mph")
    print(f"angle:  {result['launch_angle_deg']:.2f} deg")
    print(f"blobs:  {result['num_ball_images']} used "
          f"(excluded: {result['excluded_border_blobs']} border, "
          f"{result['excluded_size_blobs']} size, {result['excluded_outlier_blobs']} outlier; "
          f"consensus {result['consensus_size']})")
    print(f"scale:  {result['scale_mm_per_px']:.4f} mm/px")
    print(f"fit rms: {result['line_fit_rms_px']:.3f} px, spacing cv: {result['spacing_consistency']:.4f}")
    if result["missing_flash_detected"]:
        print("note: uneven spacing detected, corrected via minimum-spacing (likely a missed flash)")
    if result["low_confidence_spacing"]:
        print("note: uneven spacing detected but only 3 balls -- low confidence, not corrected")
