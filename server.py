"""Device-side WebSocket server speaking the Mulligan device protocol v1.1.

See docs/device-protocol.md for the full contract. The device is the
WebSocket *server*; the app connects to it as a client. This
implementation emits a `shot` on a keypress by running the real
measurement pipeline (generate.py -> detect.py) against a freshly
synthesized frame -- the camera and strobe hardware don't exist yet, but
the pipeline that will eventually process their output does, so this
exercises that pipeline rather than fabricating a shot message directly.

Run with: python server.py --serve
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid

import websockets

import detect
import generate

PROTOCOL_VERSION = {"major": 1, "minor": 1}
FIRMWARE_VERSION = "0.1.0"

# Exactly what this device can actually measure -- not spin, not start
# line, regardless of what the protocol itself supports. See
# docs/device-protocol.md's "What the device must never send".
CAPABILITIES = ["ballSpeed", "launch"]

STATUS_INTERVAL_S = 3.0

# Range swept for the synthetic shot: realistic golf launch conditions,
# matching test.py's own accuracy-sweep range.
SHOT_SPEED_RANGE_MPH = (60.0, 180.0)
SHOT_ANGLE_RANGE_DEG = (8.0, 30.0)
SHOT_NUM_FLASHES_CHOICES = (3, 4)


def measure_synthetic_shot(max_attempts=10):
    """Run the real detection pipeline against a freshly generated,
    synthetic frame standing in for the camera+strobe hardware. Returns
    (ball_speed_mph, launch_angle_deg), or None if every attempt in this
    batch either doesn't fit in frame or fails detection (both rare, and
    retried rather than surfaced -- this is a demo shot generator, not the
    measurement itself)."""
    for _ in range(max_attempts):
        speed_mph = random.uniform(*SHOT_SPEED_RANGE_MPH)
        angle_deg = random.uniform(*SHOT_ANGLE_RANGE_DEG)
        num_flashes = random.choice(SHOT_NUM_FLASHES_CHOICES)
        try:
            image, truth = generate.generate_frame(speed_mph, angle_deg, num_flashes=num_flashes)
        except ValueError:
            continue  # this speed/angle/flash-count combo doesn't fit in frame; try another draw
        result = detect.measure(image, truth["strobe_interval_ms"])
        if result is not None:
            return result["ball_speed_mph"], result["launch_angle_deg"]
    return None


class DeviceServer:
    """Holds the per-boot state (bootId, seq counter) and the set of
    currently-connected app clients, and implements the protocol."""

    def __init__(self, device_id):
        self.device_id = device_id
        self.boot_id = str(uuid.uuid4())
        self.seq = 0
        self.clients = set()

    def _next_seq(self):
        seq = self.seq
        self.seq += 1
        return seq

    def _hello_message(self):
        return {
            "type": "hello",
            "protocolVersion": PROTOCOL_VERSION,
            "deviceId": self.device_id,
            "bootId": self.boot_id,
            "firmwareVersion": FIRMWARE_VERSION,
            "capabilities": CAPABILITIES,
        }

    @staticmethod
    def _status_message(ready=True, detail=None):
        msg = {"type": "status", "ready": ready}
        if detail is not None:
            msg["detail"] = detail
        return msg

    def _shot_message(self, ball_speed_mph, launch_deg):
        # Send only what was measured: ball speed, launch angle, timestamp.
        # No spin, no club, no carry distance -- this device didn't
        # measure any of those, and a value it computed instead of
        # measured is indistinguishable from a real measurement once it's
        # on the wire. See docs/device-protocol.md's "What the device
        # must never send".
        return {
            "type": "shot",
            "seq": self._next_seq(),
            "ballSpeedMph": ball_speed_mph,
            "launchDeg": launch_deg,
            "timestamp": int(time.time() * 1000),
        }

    async def _broadcast(self, message):
        if not self.clients:
            print("[server] no app connected -- message not delivered")
            return
        data = json.dumps(message)
        await asyncio.gather(*(ws.send(data) for ws in list(self.clients)), return_exceptions=True)

    async def emit_shot(self):
        result = measure_synthetic_shot()
        if result is None:
            print("[server] synthetic frame failed detection -- try again")
            return
        speed_mph, angle_deg = result
        msg = self._shot_message(speed_mph, angle_deg)
        print(f"[server] shot seq={msg['seq']}: {speed_mph:.1f} mph @ {angle_deg:.1f} deg")
        await self._broadcast(msg)

    async def _status_loop(self):
        while True:
            await asyncio.sleep(STATUS_INTERVAL_S)
            await self._broadcast(self._status_message(ready=True))

    async def _handle_client(self, websocket):
        self.clients.add(websocket)
        print(f"[server] app connected ({websocket.remote_address}); bootId={self.boot_id}")
        try:
            await websocket.send(json.dumps(self._hello_message()))
            await websocket.send(json.dumps(self._status_message(ready=True)))
            async for raw in websocket:
                await self._handle_message(websocket, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print("[server] app disconnected")

    @staticmethod
    async def _handle_message(websocket, raw):
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            print(f"[server] ignoring malformed frame: {raw!r}")
            return
        msg_type = msg.get("type")
        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong", "nonce": msg.get("nonce")}))
        elif msg_type == "pong":
            pass  # liveness reply to a ping this device doesn't currently send
        else:
            print(f"[server] received unhandled message type: {msg_type!r}")

    async def _keypress_loop(self):
        loop = asyncio.get_event_loop()
        print("Press Enter (or 's') to emit a shot, 'q' to quit.")
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return  # stdin closed (e.g. piped input ran out)
            command = line.strip().lower()
            if command == "q":
                print("[server] shutting down")
                return
            await self.emit_shot()

    async def serve(self, host, port):
        async with websockets.serve(self._handle_client, host, port):
            print(f"[server] listening on ws://{host}:{port} "
                  f"(deviceId={self.device_id}, bootId={self.boot_id})")
            await self._keypress_loop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True,
                         help="start the device WebSocket server")
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default: 8080)")
    parser.add_argument("--device-id", default="mulligan-device-01", help="stable device identifier")
    args = parser.parse_args()

    server = DeviceServer(device_id=args.device_id)
    asyncio.run(server.serve(args.host, args.port))


if __name__ == "__main__":
    main()
