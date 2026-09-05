# Mulligan Device

A DIY golf launch monitor. This repo measures ball speed and vertical launch angle from a single
strobe photograph. It does not simulate ball flight, does not know about golf clubs, and does not
compute carry distance — a separate app owns that.

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

Everything runs on a laptop against generated images. No camera capture, no GPIO, no strobe hardware
control, no networking, no ball flight simulation — those come later.

## Running it

```
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy

python generate.py   # writes frame.png / frame.json for a single example shot
python detect.py frame.png 2.9   # measure a single frame (image path, strobe interval in ms)
python test.py       # run the full accuracy sweep; exits non-zero on failure
```

## How the scale is derived

The naive approach — assume the ball is exactly 0.6 m from the lens and use a fixed mm-per-pixel
constant — breaks badly if the ball is placed a few centimeters off that assumed distance: a 5 cm
error is an ~8% scale error, i.e. ~12 mph of speed error on a driver swing.

Instead, `detect.py` derives the scale from the ball's known real diameter (42.67 mm) and its measured
diameter in pixels, averaged across every ball image in the frame. This self-calibrates on every shot
and makes the camera-to-ball distance irrelevant to the measurement.
