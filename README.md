# Mulligan Device

A DIY golf launch monitor. This repo measures ball speed and vertical launch angle from a single
strobe photograph, and reports them to the companion app over the device protocol
(`docs/device-protocol.md`). It does not simulate ball flight, does not know about golf clubs, and
does not compute carry distance — a separate app owns that.

## Status

**Real:** the detection pipeline (`detect.py`) and everything in it — thresholding, size-consistency
filtering, the spacing-aware RANSAC line fit, the per-blob perpendicular-extent scale derivation, the
missing-flash correction, and lens-distortion correction (`calibrate.py` + `detect.py`'s optional
`calibration` argument). The accuracy numbers in this README and in `test.py` are measured, not
aspirational. The device protocol server (`server.py`) is a real, working WebSocket server speaking
real protocol v1.1 — verified end-to-end against the actual app (tag `e2e`): `hello`/`bootId` accepted,
a `shot` produced a ball flight with correct provenance.

**Synthetic:** every frame `detect.py` has ever measured, including the one behind that end-to-end
verification, was drawn by `generate.py`, not captured by a camera. There is no camera. There is no
strobe. There is no GPIO. `calibrate.py` has recovered known distortion coefficients only from a
synthetic checkerboard, never a printed one photographed by a real lens. `server.py` calling
`generate.generate_frame()` on a keypress is a stand-in for hardware that doesn't exist yet, not a
simulation of one — the detection and protocol code downstream of that frame is the exact code that
will run against a real photograph.

Next work here is hardware bring-up, not software: build the camera/strobe rig, get one real frame
(see `docs/hardware-readiness.md`), and see how much of the above still holds.

## The strobe concept

A camera sits about 2 feet to the side of a golf ball. When the club swings through, the camera opens
its shutter and holds it open while an infrared LED strobe fires several times in quick succession.
Each flash freezes the ball at a different point in its first couple of feet of flight, so a single
photograph ends up containing four to six images of the same ball in a rising line.

From that one photograph:

- **Ball speed** comes from the spacing between consecutive ball images, divided by the known time
  between flashes.
- **Launch angle** comes from the slope of the line those ball images form.

The ball travels in a straight line at roughly constant speed over the ~10 ms a frame covers — close
enough to true over two feet of flight that gravity and drag don't matter yet.

## Files

- `generate.py` — draws synthetic strobe photographs from known ball speed, launch angle, and flash
  count, and writes out the matching ground truth (PNG + JSON).
- `detect.py` — takes a photograph and the strobe interval, and measures ball speed and launch angle
  from it.
- `test.py` — generates frames across a realistic speed/angle range, runs the detector against each,
  and compares measured values against the ground truth.
- `server.py` — the device side of `docs/device-protocol.md`: a WebSocket server the app connects to,
  emitting a `shot` (measured by the real generate.py/detect.py pipeline against a synthetic frame) on
  a keypress.
- `calibrate.py` — recovers camera matrix and lens distortion coefficients from checkerboard photos;
  see `docs/calibration.md`.
- `docs/device-protocol.md` — the app/device contract. Canonical copy lives in the app repo; this is a
  mirror kept for local reference. See the file's own header before editing it.
- `docs/calibration.md` — what to print and photograph, and how to run `calibrate.py`.
- `docs/hardware-readiness.md` — what to revisit once real frames exist: tunable constants picked
  against synthetic images, assumptions the generator makes that reality won't, and the recommended
  first real test.

Everything runs on a laptop. No camera capture, no GPIO, no strobe hardware control, no ball flight
simulation — those come later. The device protocol server is real, but its shots are synthetic until
the camera exists.

## Running it

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # opencv-python, numpy, websockets

python generate.py   # writes frame.png / frame.json for a single example shot
python detect.py frame.png 2.2   # measure a single frame (image path, strobe interval in ms)
python test.py       # run the full accuracy sweep; exits non-zero on failure
```

`generate_frame()` also accepts `noise_sigma` (Gaussian sensor noise), `flash_duration_us` (motion
smear from a non-instantaneous flash), `ir_falloff` (inverse-square dimming with distance from the
strobe), `laser_dot` / `mat_reflection` / `clubhead_edge` (distractors a real range frame can contain),
`missing_flash_index` (a strobe pulse that didn't fire), `dist_coeffs`/`camera_matrix` (lens distortion,
applied as a whole-frame warp after drawing), and `depth_profile` (a per-flash relative distance factor,
for a flight where the ball moves toward or away from the camera) to exercise the detector against
harder, more realistic frames.

`laser_dot` and an isolated round `mat_reflection` both survive detect.py's basic per-blob filters
(area, circularity, border) -- neither checks a candidate's size against the expected ball diameter.
What actually rejects them is two frame-level checks in `measure()`: a relative size-consistency filter
(all genuine ball images in one frame are the same physical object, so they cluster tightly in size
regardless of what that size actually is) and a RANSAC-style line fit that keeps only the largest
mutually collinear, evenly-spaced consensus set. The two are complementary -- a reflection that happens
to be ball-sized still gets caught by the line fit for not being on the flight line, and something
small enough to be size-filtered never reaches the line fit at all. `test.py`'s `check_distractors`
shows the basic filters alone letting these through; `check_distractors_combined` shows the full
pipeline (all three distractors, plus noise and falloff, at once) handling them.

## How the scale is derived

The naive approach — assume the ball is exactly 0.6 m from the lens and use a fixed mm-per-pixel
constant — breaks badly if the ball is placed a few centimeters off that assumed distance: a 5 cm
error is an ~8% scale error, i.e. ~12 mph of speed error on a driver swing.

Instead, `detect.py` derives the scale from the ball's known real diameter (42.67 mm), fitting the
flight line through the ball centres first and then measuring each blob's extent *perpendicular* to
that line rather than its raw diameter or area. This self-calibrates on every shot (the camera-to-ball
distance stops mattering) and stays accurate even when motion smear stretches each ball image along
its direction of travel, since smear doesn't affect the perpendicular extent.

The ball doesn't stay at a constant distance from the camera in reality — it's moving in three
dimensions, so its apparent size shifts across the frame, and each ball's own diameter already tells you
its own distance. `detect.py` derives a scale *per blob* rather than one global mean, and converts each
gap between consecutive balls using the average of its two endpoint scales, so real depth change is
absorbed instead of averaged away. `measure()`'s result includes `per_blob_scales_mm_per_px` and
`scale_spread_mm_per_px` — a spread that's a useful diagnostic on real hardware, since a large one means
either genuine depth change or a detection problem. A single blob whose scale is wildly out of line with
the rest (`scale_outlier_blobs`) is treated as a bad measurement rather than real depth change, and
falls back to the median scale for that blob alone.

Segmentation uses an adaptive, descending sequence of thresholds rather than one fixed cutoff, so a
ball dimmed by IR falloff (farther from the strobe) can still be found even when a brighter, closer
ball in the same frame would saturate a threshold tuned for the dim one. The threshold sequence is
derived from the frame's own background/noise statistics (median and MAD), not blind halving, so it
can't itself dip below the ambient background and threshold the whole frame to white.

`measure()`'s result includes `excluded_border_blobs`, `excluded_size_blobs`, `excluded_outlier_blobs`,
and `consensus_size` -- together they say *where* blobs were lost (edge clipping, size inconsistency,
or failing the line-fit consensus), which on real hardware is what distinguishes a bad reading caused
by detection from one caused by measurement.

The line fit itself checks two things, not one: candidate ball images must be both collinear *and*
evenly spaced (gaps consistent with integer multiples of one common flash interval, so a genuine missed
flash still passes). Collinearity alone isn't enough -- a distractor that happens to sit on the flight
line (a laser dot marking the address position, say) would pass a collinearity-only test trivially;
it's the spacing check that catches it, since its position along the line rarely lines up with the
flash timing.

## Camera calibration

A cheap wide-FOV lens shows real barrel distortion toward the edges, and since ball images span nearly
the full frame width, that distortion corrupts the spacing between them directly. `calibrate.py` recovers
a camera matrix and distortion coefficients from checkerboard photos (see `docs/calibration.md` for what
to print and photograph); `detect.py`'s optional `calibration` argument undistorts a frame before
detection using the result. Both are fully testable synthetically: `generate_frame()` and
`generate_checkerboard_frame()` accept the same `dist_coeffs`/`camera_matrix`, so `test.py` generates a
distorted checkerboard, confirms `calibrate.py` recovers the known coefficients, then generates a
distorted ball frame and confirms undistorting it measurably reduces speed error.

## Running the device server

```
python server.py --serve                    # listens on ws://0.0.0.0:8080 by default
python server.py --serve --port 9000        # non-default port
python server.py --serve --device-id my-pi  # stable id the app sees across reboots
```

The server prints its `bootId` on startup and waits for an app to connect. Once connected, press Enter
(or `s`) in the server's terminal to emit one `shot` -- generated and measured by the real
generate.py/detect.py pipeline against a fresh synthetic frame, not fabricated -- to every connected
app. Press `q` to shut down.

**The loop, end to end:** start `python server.py --serve` on this machine, point the app's Device
source at `ws://<this machine's address>:8080` (`ws://localhost:8080` if the app runs on the same
machine) and connect, then press a key in the server's terminal and watch a ball fly on the app's hole.
Killing the server should show a clear disconnected state in the app without ending its round;
restarting it should let the app reconnect (same `bootId` if the process wasn't restarted, a new one if
it was -- see `docs/device-protocol.md`'s notes on `bootId`/`seq` dedup).
