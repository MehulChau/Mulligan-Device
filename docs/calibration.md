# Camera calibration

Why this matters: the 6mm lens on this sensor gives roughly a 55-degree field of
view — wide enough that a cheap lens shows real barrel distortion toward the
edges. Ball images span nearly the full frame width, so that distortion
corrupts the spacing between them directly, which is the measurement. At a
realistic ~3% distortion near the edge, that's comparable to a 3% speed
error — larger than every other error source in this system combined. Do
this once per camera/lens, and again if the lens is refocused or replaced.

## What to print

A standard black-and-white checkerboard, **10 x 7 squares** (which OpenCV
counts as **9 x 6 internal corners** — the corners where 4 squares meet,
which is what `calibrate.py`'s default `--pattern-cols 9 --pattern-rows 6`
expects). Print it as large as will comfortably fit your printer and stay
flat — bigger is better as long as the board doesn't sag or curl, since any
warp in the physical board becomes a false "distortion" signal in the fit.

- **Square size:** measure your printed squares with a ruler and pass the
  real number to `--square-size-mm` — don't trust the print dialog's scaling.
  25mm squares (the script's default) print comfortably on a single
  US-Letter/A4 page.
- **Mount it on something rigid** — foam board or a clipboard — a sheet of
  paper alone will not stay flat enough.

## What to photograph

You need photos that vary in two ways, not one:

1. **Position across the frame** — some boards centered, some pushed toward
   each edge and corner. Distortion is worst at the edges, so a calibration
   built from twenty photos of a board in the middle of the frame is
   confidently wrong exactly where it matters. `calibrate.py` warns if the
   detected corners, pooled across every photo, never reach near an edge.
2. **Angle relative to the camera** — tilt the board left, right, up, and
   down relative to straight-on, not just move it side to side. This is the
   one that's easy to forget and that actually matters most: a set of
   photos that are all fronto-parallel (just moved around or closer/further)
   under-constrains the fit. The focal length and the distortion trade off
   against each other with no way for the math to tell them apart, and
   `cv2.calibrateCamera` will converge on a confident, self-consistent, and
   **badly wrong** answer with no warning. Real perspective from real tilt
   is what breaks that ambiguity.

**In practice:** take 15-20 photos. For each, hold the board at a different
position (try the centre, all 4 corners, and the 4 edge midpoints of the
frame at least once) *and* a different tilt (try flat-on, and then tilted
roughly 20-30 degrees in each of up/down/left/right at least a few times).
Vary the distance a little too (fill roughly a third to two-thirds of the
frame across your photos) — different distances help the same way different
angles do. Keep the board in focus and well and evenly lit in every shot;
motion blur or a washed-out/shadowed board will hurt corner detection more
than a suboptimal pose will.

## Running it

```
python calibrate.py <directory-of-photos> \
    --pattern-cols 9 --pattern-rows 6 \
    --square-size-mm 25 \
    --out calibration.json
```

It will:

- Find the checkerboard in each photo and refine each corner to sub-pixel
  precision.
- Warn if you gave it fewer than 10 usable photos, or if a photo's board
  wasn't found at all (printed per-file, with the reason).
- Warn if the detected corners don't reach near one or more frame edges.
- Run the calibration and print the **reprojection error** (in pixels) and,
  in physical terms, **how far a point near the frame edge moves when
  undistorted** — that second number is the one that tells you whether any
  of this was worth doing. A few pixels means the lens is already quite
  clean; tens of pixels means it very much was not.
- **Refuse to write `calibration.json` if the reprojection error exceeds
  0.5px** (`--max-reprojection-error` to change it). A bad calibration is
  worse than none: it silently distorts every subsequent measurement instead
  of leaving them alone. If it refuses, retake photos with sharper focus,
  better and more even lighting, and a board that doesn't have printing
  banding or paper warp.

## Using the result

Pass the calibration file (or the loaded dict) to `detect.measure()`:

```python
result = detect.measure(image, strobe_interval_ms, calibration="calibration.json")
```

`result["calibration_applied"]` and `result["calibration_source"]` record
whether a calibration was used and which one, so a measurement can always be
traced back to the calibration that produced it. Passing nothing
(the default) skips undistortion entirely — every existing synthetic test
in this repo assumes a distortion-free pinhole image and is unaffected.

## Testing this without a camera

`generate.py`'s `generate_checkerboard_frame()` and `generate_frame()` both
accept the same `dist_coeffs`/`camera_matrix`, so the whole pipeline —
distort a synthetic checkerboard, recover the coefficients with
`calibrate.py`, distort a synthetic ball frame, and confirm undistorting it
with the recovered calibration reduces speed error — is exercised end to end
by `test.py` with no hardware at all. See `check_calibration_recovers_distortion`
and `check_undistortion_reduces_speed_error` there for the exact numbers this
repo has actually verified.
