# Hardware readiness

Everything in this repo has been verified against synthetic images only.
This is the list of what to revisit once real frames exist — written down
now, while the reasoning for each choice is fresh, rather than
reconstructed later when something on real hardware doesn't match.

## Tunable constants picked against synthetic images

All of these were chosen (or empirically tuned) against clean or
syntheticaly-noised images. None have been checked against anything a real
sensor and lens actually produce.

- **`_adaptive_thresholds`'s threshold sequence** (`detect.py`) — derived
  from the frame's own median/MAD, floored at background + 3σ, stepping in
  multiples of {10, 6, 4, 3}σ. The MAD-based σ estimator assumes the
  background is the majority of the frame and roughly Gaussian; a real
  background (mat texture, ambient light gradient, sensor fixed-pattern
  noise) may not be either.
- **`SIZE_CONSISTENCY_BAND = (0.6, 1.6)`** — the band a blob's size must
  fall within, relative to the modal detected size, to survive. Chosen to
  be "generous enough to pass a real ball under any modeled smear/noise,
  tight enough to reject a laser dot off by 10x+." Never tested against a
  real ball's actual apparent-size variation (real optical blur, sensor
  noise, and JPEG-like compression artifacts, if any compression is in the
  capture path, all inflate or distort a blob's measured size in ways the
  synthetic anti-aliased disk model doesn't).
- **`SPACING_RANSAC_TOLERANCE = 0.08`** — the relative-to-base-gap
  tolerance for the RANSAC line fit's spacing check. Currently budgeted for
  ~0.3% aerodynamic drag deceleration and ~0.3% centroid/moment noise on a
  ~333px base gap, both comfortably inside it. The unknown is real strobe
  timing jitter, which doesn't exist yet because the strobe is simulated at
  an exactly regular interval. **If real timing jitter approaches or
  exceeds ~8% of the base gap, valid shots will start being rejected here
  as spacing-inconsistent** — this is the first constant to revisit once
  real frames exist to measure actual jitter against.
  - A second, related pressure on this same constant, found while
    implementing per-blob scale: a flight where the ball's distance from
    the camera changes measurably (see below) produces genuinely uneven
    *pixel* spacing even at constant real-world speed. A ~9% depth change
    across a 4-flash shot was about the most this tolerance could absorb
    before the line fit started (correctly, given the pixel-only spacing
    check) treating the depth-varying real balls as spacing-inconsistent
    and trimming one. Real depth change and real timing jitter both spend
    out of the same 8% budget, and there's currently no way to tell which
    one used it up.
- **The circularity cutoff (0.7)** (`detect.py`, in `_find_blobs`) — chosen
  so a clean synthetic disk (circularity ≈0.90-0.93 after rasterization)
  passes with margin and a synthetic triangle/elongated smudge (≈0.49-0.57)
  fails with margin. A real ball image's circularity under real motion blur,
  real lens softness, and real sensor noise has never been measured.
- **`RANSAC_INLIER_THRESHOLD_PX = 5.0`** — the perpendicular-distance
  budget for a blob to count as on the flight line. Chosen as "small
  relative to typical spacing, generous relative to expected centroid
  noise," but never validated against real lens distortion (uncorrected,
  a real lens bows a straight flight path into a very slight curve, which
  this threshold has to absorb) or real vibration in the camera mount.
- **Calibration's `MAX_REPROJECTION_ERROR_PX = 0.5`** default and
  `CALIB_FIX_K3` — reasonable defaults for a ~55-degree-FOV lens, unvalidated
  against how well a real, hand-photographed checkerboard set actually
  converges (synthetic checkerboards are photographed with perfect focus,
  perfect exposure, and pixel-perfect corner rendering; a real photo set
  will have all three real-world imperfections at once).

## Assumptions the synthetic generator makes that reality won't

- **A uniform-brightness disk, not a sphere lit from one side.** A golf
  ball under a single IR strobe is a lit sphere, not a flat, evenly-bright
  disk — real ball images will have a brightness gradient across their
  face. This may pull the intensity-weighted centroid slightly toward the
  lit side, and *not identically* for each exposure if the strobe's angle
  to the ball changes even slightly as it moves through the frame,
  introducing a per-flash centroid bias that this repo's noise model
  (symmetric, spatially uniform) cannot represent or catch.
- **No lens vignetting.** Real lenses dim toward the corners on top of
  whatever IR falloff the strobe geometry already causes; `ir_falloff`
  models the latter but not the former. The two would compound in a real
  frame.
- **Constant camera-to-ball distance, except where `depth_profile`
  explicitly overrides it for testing.** The default (and every existing
  test except the depth-varying one) assumes flat, 2D motion parallel to
  the sensor. Per-blob scale (this milestone) exists specifically because
  real 3D motion doesn't hold to that.
- **No ambient infrared.** See below — this is the big one.

## Ambient IR is the biggest practical risk, and it's a hardware fix

Sunlight is full of 850nm light. Outdoors in daylight, ambient IR can swamp
the strobe entirely: the near-black background this entire detection
pipeline assumes (and every noise/threshold constant above is tuned
against) becomes a bright, textured one instead. No software fix in this
repo — not the adaptive threshold, not the size filter, not RANSAC — was
designed for a background that's brighter than some genuine ball images
under falloff, because that scenario inverts the basic assumption
(strobe-lit foreground on a dark background) the whole pipeline is built on.

**An 850nm bandpass filter on the lens is close to mandatory for outdoor
use.** This is a hardware decision to make before the first outdoor test,
not a threshold to retune afterward.

## The first real test

**Roll a ball across a table indoors, do not swing a golf club at it.** A
rolled ball gives a real frame — real sensor noise, real lens softness,
real (mild) motion blur, a real if imperfect background — at low stakes and
zero risk to the not-yet-calibrated hardware, and it's the first time
anything in this project will have been checked against something outside
its own simulation.

Once that frame exists:

1. Read off its approximate speed, angle, and distance-to-camera by hand
   (a tape measure and a stopwatch is plenty of precision for this).
2. Generate a synthetic frame with `generate.py` at those same approximate
   parameters.
3. Compare the two frames directly — background level, ball sharpness and
   apparent size, noise texture, anything that looks structurally different
   rather than just numerically different.

That comparison is what turns the items in this document from guesses into
a prioritized list: whichever assumption the real frame violates most
visibly is the one to fix first, before ever pointing the rig at an actual
golf swing.
