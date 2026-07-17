# SO-100 / SO-101 arm support

Self-contained vendor of the [LeRobot](https://github.com/huggingface/lerobot) 0.4.4
hardware stack for SO-100/SO-101 arms — Feetech STS3215 motor bus, follower robot,
leader teleoperator, OpenCV cameras — plus CLI scripts for the full data-collection
workflow. Recording writes **OpenTau's native `LeRobotDataset` (v2.1)** via
`opentau.datasets`, so recorded datasets train directly with `opentau-train`.

## Install

```bash
uv sync --extra so101
```

Calibration files are read/written at `~/.cache/huggingface/lerobot/calibration`
(identical to upstream LeRobot — existing calibrations are picked up as-is).

## Workflow

All commands are also available as `opentau-so101-*` console scripts.

### 0. Find the serial port of each arm (unplug/replug prompt)

```bash
python -m opentau.scripts.so101.find_port
```

Prefer the stable `/dev/serial/by-id/...` paths over `/dev/ttyACM*` — ACM numbering
swaps across reboots.

### 1. Set motor IDs (only for freshly assembled arms; one motor connected at a time)

```bash
python -m opentau.scripts.so101.setup_motors \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

### 2. Calibrate (once per arm)

```bash
python -m opentau.scripts.so101.calibrate \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower

python -m opentau.scripts.so101.calibrate \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
```

Follow the prompts: move to mid-range, ENTER, sweep every joint through its full
range, ENTER.

### 3. Find cameras

```bash
python -m opentau.scripts.so101.find_cameras --save-dir /tmp/so101_cams
```

### 4. Teleoperate (leader drives follower)

```bash
python -m opentau.scripts.so101.teleoperate \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
```

Add `--robot.cameras='{...}'` and `--display_data=true` for a live rerun view.

### 5. Record a dataset

```bash
python -m opentau.scripts.so101.record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
    --robot.cameras='{
        "front": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader \
    --dataset.repo_id=TensorAuto/my_task \
    --dataset.single_task="put the red chip bag into the white bin" \
    --dataset.num_episodes=50 \
    --dataset.push_to_hub=false
```

Keyboard controls while recording: **right arrow** ends the episode, **left arrow**
re-records it, **escape** stops after the current episode. `--resume=true` appends
episodes to an existing local dataset. The recorded dataset carries
`robot_type=so_follower` and `control_mode=joint` in `meta/info.json` and loads
directly in OpenTau training configs.

## Hardware notes (from the reference rig)

- Two raw-YUYV USB cameras exhaust USB2 isochronous bandwidth: the second camera
  opens but never delivers frames. Give the MJPEG-capable camera
  `"fourcc": "MJPG"` in its config and keep the YUYV-only camera raw.
- If connect fails with "found motor list {1..5}" (gripper missing), the gripper
  servo latched an overload flag: free the jaws and power-cycle the arm.
- Serial access needs `dialout` group membership (or an ACL) on the `/dev/ttyACM*`
  nodes.

## Layout

```
so101/
├── calibrate.py / teleoperate.py / record.py      CLI scripts
├── setup_motors.py / find_port.py / find_cameras.py
├── motors/            Feetech STS3215 bus (MotorsBus, FeetechMotorsBus, tables)
├── cameras/           Camera ABC + OpenCV camera
├── robots/            Robot ABC + SO100/SO101 follower
├── teleoperators/     Teleoperator ABC + SO100/SO101 leader
├── constants.py       calibration paths (upstream-compatible)
├── errors.py / decorators.py / types.py / utils.py
```

Vendored from LeRobot 0.4.4 (Apache-2.0) with imports rewritten to
`opentau.scripts.so101.*`; the processor package is replaced by dict aliases
(`types.py`) and dataset writing is delegated to `opentau.datasets`.
