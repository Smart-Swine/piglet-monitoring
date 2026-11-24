"""
D456 recorder configured for the ``pi`` user on the new Raspberry Pi.

The two cameras are deliberately independent:

* D456: its pipeline stays open continuously with the emitter OFF while idle.
  At :00/:30 it enables the emitter, detects the first emitter-ON frameset,
  saves its preceding OFF-emitter IR image, then records RGB, depth and IR for
  a full 20 seconds at 5 FPS.
* Arducam, VDA export, and scheduled high-rate clips are disabled by default
  for this deployment.

Press q in either preview window, or Ctrl+C in the terminal, to stop safely.
"""

import datetime as dt
import json
import multiprocessing as mp
import os
import signal
import threading
import time
import warnings
from queue import Empty, Full, Queue

import cv2
import numpy as np
import pyrealsense2 as rs
import scipy.io

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_ROOT = os.environ.get(
    "D456_DATA_ROOT",
    os.path.join(os.path.expanduser("~"), "Pictures", "farrow_data"),
)
CALIBRATION_DIR_NAME = "CALIBRATION"
HEALTH_STATUS_FILE = os.environ.get(
    "D456_HEALTH_STATUS_FILE",
    os.path.join(SCRIPT_DIR, "recorder_camera_health.status"),
)

ARDUCAM_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Arducam_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0"
)

ARDUCAM_WIDTH = 640
ARDUCAM_HEIGHT = 480

RS_WIDTH = 848
RS_HEIGHT = 480

D456_NORMAL_SAVE_FPS = 5
D456_HIGH_RATE_SAVE_FPS = 30
HIGH_RATE_INTERVAL_HOURS = 3
D456_RECORD_SECONDS = 20

# True: run the independent Arducam and save AR/*.avi files.
# False: do not open, monitor, display, or save the Arducam.
ENABLE_ARDUCAM = False

# True: the first clip every three hours is saved at 30 FPS.
# False: every D456 RGB/IR/depth clip is saved at 5 FPS.
ENABLE_HIGH_RATE_CLIPS = False

# True: save one DM point-cloud MAT file for every 30-second recording slot.
# False: save DM only at 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00
# and 21:00. The next day's 00:00 is the requested 24:00 slot.
SAVE_DM_EVERY_CLIP = True

# The D456 must stream at 30 FPS whenever high-rate clips are enabled. Turning
# the feature off restores the original 5 FPS device stream and lowers load.
D456_SOURCE_FPS = (
    D456_HIGH_RATE_SAVE_FPS
    if ENABLE_HIGH_RATE_CLIPS
    else D456_NORMAL_SAVE_FPS
)

# Every clip belongs to an absolute :00/:30 wall-clock slot.  The RealSense
# pipeline stays running between clips; only the emitter is switched off.
RECORDING_SLOT_SECONDS = 30

RGB_SAVE_FPS = 30
RGB_SEGMENT_SECONDS = 30
RGB_FILE_FRAMES = RGB_SAVE_FPS * RGB_SEGMENT_SECONDS  # 900 when healthy

CAMERA_RETRY_SECONDS = 2.0
D456_FRAME_STALL_SECONDS = 2.0
ARDUCAM_METADATA_STALL_SECONDS = 2.0

# This is handled by a parent-process watchdog, including native calls that
# never return.
ARDUCAM_HARD_BLOCK_SECONDS = 5.0
D456_HARD_BLOCK_SECONDS = 15.0
PROCESS_STARTUP_TIMEOUT_SECONDS = 30.0

# During every idle interval the camera streams with the emitter OFF. After
# the boundary the first frame whose hardware metadata reports emitter ON is
# used as the first video frame; its immediately preceding IR frame is saved.
EMITTER_DETECTION_TIMEOUT_SECONDS = 4.0
DEPTH_COLORMAP_ALPHA = 0.125

# Optional VDA export. Keep this False when VDA data is not needed; all other
# recordings remain enabled. Change it to True to save RGB + aligned depth.
RECORD_VDA = False

# VDA always contains at most 100 pairs. That spans 20 seconds in an ordinary
# 5 FPS clip and about 3.33 seconds in a scheduled 30 FPS clip.
VDA_NUM_FRAMES = 100
VDA_JPEG_QUALITY = 95

FOURCC = cv2.VideoWriter_fourcc(*"XVID")


class EmitterMetadataUnavailableError(RuntimeError):
    """The active D456 stream does not expose per-frame emitter metadata."""


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

stop_event = threading.Event()

rgb_lock = threading.Lock()
rgb_latest_frame = None
rgb_latest_sequence = 0
rgb_latest_capture_wall = 0.0

preview_lock = threading.Lock()
depth_preview = None
rgb_preview = None

# One complete clip can be queued without forcing RealSense capture to wait for
# JPEG/PNG compression. The worker normally drains this concurrently.
vda_queue = Queue(maxsize=128)
dm_queue = Queue(maxsize=4)
ir_snapshot_queue = Queue(maxsize=8)


# ---------------------------------------------------------------------------
# GPIO is optional, so the script can still be tested away from the Pi.
# ---------------------------------------------------------------------------

class DummyLED:
    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass


yellow = DummyLED()
green = DummyLED()
red = DummyLED()


def initialize_leds():
    """Initialize GPIO only in the main process, never in camera subprocesses."""
    global yellow, green, red
    try:
        from gpiozero import LED
        from gpiozero.pins.lgpio import LGPIOFactory

        gpio_factory = LGPIOFactory()
        yellow = LED(17, pin_factory=gpio_factory)
        green = LED(22, pin_factory=gpio_factory)
        if os.environ.get("RECORDER_MONITOR_CONTROLS_RED") != "1":
            red = LED(27, pin_factory=gpio_factory)
    except Exception as exc:
        print("GPIO unavailable; LEDs disabled:", exc)
        yellow = DummyLED()
        green = DummyLED()
        red = DummyLED()

    yellow.on()
    green.off()
    red.on()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def ensure_directories():
    health_directory = os.path.dirname(HEALTH_STATUS_FILE)
    if health_directory:
        os.makedirs(health_directory, exist_ok=True)

    directory_names = [
        "DM",
        "RGB",
        "DP",
        "IR",
        "IR_SN",
        CALIBRATION_DIR_NAME,
    ]
    if ENABLE_ARDUCAM:
        directory_names.append("AR")
    if RECORD_VDA:
        directory_names.append("VDA_dataset")

    for name in directory_names:
        os.makedirs(os.path.join(BASE_ROOT, name), exist_ok=True)


def write_health_status(is_healthy, rgb_healthy, realsense_healthy):
    """Publish camera health for monitor.py without sharing GPIO ownership."""
    temporary_path = HEALTH_STATUS_FILE + ".tmp"
    line = (
        f"{time.time():.3f} "
        f"{'OK' if is_healthy else 'FAULT'} "
        f"arducam_enabled={int(ENABLE_ARDUCAM)} "
        f"rgb={int(rgb_healthy)} realsense={int(realsense_healthy)}\n"
    )
    try:
        with open(temporary_path, "w", encoding="ascii") as file:
            file.write(line)
        os.replace(temporary_path, HEALTH_STATUS_FILE)
    except Exception as exc:
        print("Health status write failed:", exc)


def health_indicator_loop(rgb_healthy, realsense_healthy):
    """Red is ON unless every enabled camera is producing fresh frames."""
    last_state = None
    while not stop_event.is_set():
        rgb_ok = bool(rgb_healthy.value) if ENABLE_ARDUCAM else True
        realsense_ok = bool(realsense_healthy.value)
        all_ok = rgb_ok and realsense_ok
        if all_ok != last_state:
            if all_ok:
                red.off()
                if ENABLE_ARDUCAM:
                    print("Camera health OK: Arducam and D456 are both streaming")
                else:
                    print("Camera health OK: D456 is streaming; Arducam is disabled")
            else:
                red.on()
                print(
                    "Camera fault: red LED ON;",
                    (
                        f"Arducam={'OK' if rgb_ok else 'FAULT'},"
                        if ENABLE_ARDUCAM
                        else "Arducam=DISABLED,"
                    ),
                    f"D456={'OK' if realsense_ok else 'FAULT'}",
                )
            last_state = all_ok
        write_health_status(all_ok, rgb_ok, realsense_ok)
        stop_event.wait(0.5)

    write_health_status(False, False, False)
    red.on()


def unique_stem(directory, wanted_stem, extension=".avi"):
    """Do not overwrite a file if the program is restarted in the same second."""
    stem = wanted_stem
    number = 1
    while os.path.exists(os.path.join(directory, stem + extension)):
        stem = f"{wanted_stem}_{number:02d}"
        number += 1
    return stem


def make_writer(path, fps, frame_size, is_color=True):
    writer = cv2.VideoWriter(
        path,
        FOURCC,
        fps,
        frame_size,
        isColor=is_color,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def release_writers(writers):
    for writer in writers.values():
        try:
            writer.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Independent Arducam capture
# ---------------------------------------------------------------------------

def arducam_capture_process(shared_frame, shared_sequence, heartbeat, process_stop):
    """Child process: direct V4L2 capture with sequence/timestamp checks."""
    from v4l2_sequence_capture import V4L2SequenceCapture

    shared_image = np.frombuffer(shared_frame, dtype=np.uint8).reshape(
        ARDUCAM_HEIGHT, ARDUCAM_WIDTH, 3
    )
    cap = None
    last_driver_sequence = None
    last_driver_timestamp = None
    metadata_stall_since = None

    while not process_stop.is_set():
        if cap is None:
            # If opening the device itself blocks, the parent sees this heartbeat
            # stop changing and kills this complete process.
            heartbeat.value = time.monotonic()
            try:
                cap = V4L2SequenceCapture(
                    ARDUCAM_DEVICE,
                    width=ARDUCAM_WIDTH,
                    height=ARDUCAM_HEIGHT,
                    fps=RGB_SAVE_FPS,
                    pixel_format="MJPG",
                    buffer_count=4,
                )
            except Exception as exc:
                print("Arducam V4L2 open failed; retrying:", exc, flush=True)
                cap = None
                heartbeat.value = time.monotonic()
                process_stop.wait(CAMERA_RETRY_SECONDS)
                continue
            print(
                "Arducam direct V4L2 capture started:",
                f"{cap.width}x{cap.height}",
                cap.pixel_format,
                flush=True,
            )
            last_driver_sequence = None
            last_driver_timestamp = None
            metadata_stall_since = None

        # The heartbeat is set immediately before the potentially blocking call.
        # If read never returns, its age exceeds ARDUCAM_HARD_BLOCK_SECONDS.
        heartbeat.value = time.monotonic()
        try:
            frame, driver_sequence, driver_timestamp = cap.read(timeout=2.0)
        except Exception as exc:
            print("Arducam V4L2 read failed; reopening:", exc, flush=True)
            cap.close()
            cap = None
            heartbeat.value = time.monotonic()
            process_stop.wait(0.2)
            continue
        now = time.monotonic()

        if frame.shape[:2] != (ARDUCAM_HEIGHT, ARDUCAM_WIDTH):
            frame = cv2.resize(
                frame,
                (ARDUCAM_WIDTH, ARDUCAM_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

        sequence_advanced = (
            last_driver_sequence is None
            or driver_sequence != last_driver_sequence
        )
        timestamp_advanced = (
            driver_timestamp <= 0
            or last_driver_timestamp is None
            or last_driver_timestamp <= 0
            or driver_timestamp > last_driver_timestamp
        )

        if not sequence_advanced or not timestamp_advanced:
            if metadata_stall_since is None:
                metadata_stall_since = now
            elif now - metadata_stall_since >= ARDUCAM_METADATA_STALL_SECONDS:
                print(
                    "Arducam V4L2 sequence/timestamp stopped; reopening",
                    flush=True,
                )
                cap.close()
                cap = None
                heartbeat.value = now
                process_stop.wait(0.2)
                continue
            # Never forward a buffer whose driver metadata did not advance.
            heartbeat.value = now
            continue
        else:
            metadata_stall_since = None

        last_driver_sequence = driver_sequence
        last_driver_timestamp = driver_timestamp

        # A lock-free sequence counter prevents the parent from copying a frame
        # while this process is halfway through writing it. Odd means writing;
        # even means the shared image is complete.
        sequence = shared_sequence.value
        if sequence % 2:
            sequence += 1
        shared_sequence.value = sequence + 1
        np.copyto(shared_image, frame)
        heartbeat.value = now
        shared_sequence.value = sequence + 2

    if cap is not None:
        cap.close()
    print("Arducam capture process stopped", flush=True)


def stop_capture_process(process, process_stop, graceful_timeout=1.0):
    """Stop gracefully when possible, then force-kill a blocked native call."""
    if process is None:
        return

    process_stop.set()
    process.join(timeout=graceful_timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)

    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=1.0)


def arducam_supervisor_loop(
    process_context,
    shared_frame,
    shared_sequence,
    heartbeat,
    camera_healthy,
    process_stop,
):
    """
    Parent watchdog: publish shared frames and restart a hard-blocked reader.

    The video writer is not restarted. It continues its 30 FPS clock using the
    last good frame until this supervisor receives a fresh frame.
    """
    global rgb_latest_frame, rgb_latest_sequence, rgb_latest_capture_wall, rgb_preview

    shared_image = np.frombuffer(shared_frame, dtype=np.uint8).reshape(
        ARDUCAM_HEIGHT, ARDUCAM_WIDTH, 3
    )
    process = None
    last_sequence = -1
    process_started_at = 0.0
    startup_heartbeat = 0.0
    process_ready = False
    last_fresh_frame_at = 0.0

    def start_process():
        nonlocal last_sequence, process_started_at, startup_heartbeat, process_ready
        nonlocal last_fresh_frame_at
        camera_healthy.value = 0
        process_stop.clear()
        shared_sequence.value = 0
        heartbeat.value = time.monotonic()
        startup_heartbeat = heartbeat.value
        child = process_context.Process(
            target=arducam_capture_process,
            args=(
                shared_frame,
                shared_sequence,
                heartbeat,
                process_stop,
            ),
            name="arducam-capture-process",
            daemon=True,
        )
        child.start()
        process_started_at = time.monotonic()
        process_ready = False
        last_sequence = -1
        last_fresh_frame_at = 0.0
        print("Arducam capture subprocess launched, PID:", child.pid)
        return child

    try:
        process = start_process()

        while not stop_event.is_set():
            now = time.monotonic()

            if not process.is_alive():
                camera_healthy.value = 0
                print(
                    "Arducam capture process exited; restarting, exit code:",
                    process.exitcode,
                )
                process.join(timeout=0.2)
                if stop_event.wait(0.2):
                    break
                process = start_process()
                continue

            if not process_ready and heartbeat.value != startup_heartbeat:
                process_ready = True

            heartbeat_age = now - heartbeat.value
            startup_age = now - process_started_at
            timed_out = (
                process_ready and heartbeat_age >= ARDUCAM_HARD_BLOCK_SECONDS
            ) or (
                not process_ready
                and startup_age >= PROCESS_STARTUP_TIMEOUT_SECONDS
            )
            if timed_out:
                camera_healthy.value = 0
                old_pid = process.pid
                print(
                    f"Arducam capture unresponsive for "
                    f"{heartbeat_age if process_ready else startup_age:.1f}s; "
                    f"terminating PID {old_pid}"
                )
                stop_capture_process(process, process_stop)

                if process.is_alive():
                    # A Linux task stuck in an uninterruptible kernel state may
                    # temporarily resist even SIGKILL. Do not open the USB device
                    # twice; wait and try cleanup again.
                    print("Arducam process has not exited yet; retrying cleanup")
                    if stop_event.wait(1.0):
                        break
                    continue

                print("Arducam blocked process ended; starting a new process")
                process = start_process()
                continue

            # Seqlock read: accept the copy only when the even sequence value is
            # unchanged before and after the memory copy.
            sequence_before = shared_sequence.value
            if (
                sequence_before > 0
                and sequence_before % 2 == 0
                and sequence_before != last_sequence
            ):
                candidate = shared_image.copy()
                sequence_after = shared_sequence.value
                if sequence_before == sequence_after and sequence_after % 2 == 0:
                    with rgb_lock:
                        rgb_latest_frame = candidate
                        rgb_latest_sequence += 1
                        rgb_latest_capture_wall = time.time()
                    with preview_lock:
                        rgb_preview = candidate
                    last_sequence = sequence_after
                    last_fresh_frame_at = now
                    camera_healthy.value = 1

            if (
                camera_healthy.value
                and now - last_fresh_frame_at >= ARDUCAM_METADATA_STALL_SECONDS
            ):
                camera_healthy.value = 0
                print("Arducam fresh frames stopped; red fault state requested")

            stop_event.wait(0.01)
    finally:
        camera_healthy.value = 0
        stop_capture_process(process, process_stop)
        print("Arducam supervisor stopped")


def arducam_writer_loop():
    """
    Save only fresh camera frames, grouped by real 30-second time windows.

    No frame is duplicated. A file affected by a camera outage therefore has
    fewer than 900 frames and a playback duration shorter than 30 seconds. This
    honestly represents missing data instead of fabricating visual continuity.
    """
    ar_dir = os.path.join(BASE_ROOT, "AR")

    # Wait for the camera's first valid frame before creating the first file.
    first_frame = None
    while not stop_event.is_set() and first_frame is None:
        with rgb_lock:
            if rgb_latest_frame is not None:
                first_frame = rgb_latest_frame
        if first_frame is None:
            stop_event.wait(0.05)

    if stop_event.is_set():
        return

    segment_start_wall = next_recording_boundary()
    while not stop_event.is_set() and time.time() < segment_start_wall:
        stop_event.wait(min(segment_start_wall - time.time(), 0.05))
    if stop_event.is_set():
        return

    segment_wall = dt.datetime.fromtimestamp(segment_start_wall)
    segment_deadline = segment_start_wall + RGB_SEGMENT_SECONDS
    next_tick = time.monotonic()
    tick_period = 1.0 / RGB_SAVE_FPS

    writer = None
    frames_written = 0
    last_written_sequence = -1

    try:
        while not stop_event.is_set():
            now = time.monotonic()
            now_wall = time.time()

            # Roll files on the real clock, even if the camera supplied no
            # frames. The next valid frame belongs to the current time window.
            while now_wall >= segment_deadline:
                if writer is not None:
                    writer.release()
                    writer = None
                    print(
                        "Arducam 30-second window saved:",
                        frames_written,
                        "real frames",
                    )
                    if frames_written < RGB_FILE_FRAMES:
                        print(
                            "Arducam data gap detected:",
                            RGB_FILE_FRAMES - frames_written,
                            "frames were not fabricated",
                        )
                else:
                    print("Arducam 30-second window had no valid frames")

                frames_written = 0
                segment_wall += dt.timedelta(seconds=RGB_SEGMENT_SECONDS)
                segment_deadline += RGB_SEGMENT_SECONDS

            delay = next_tick - now
            if delay > 0:
                stop_event.wait(min(delay, 0.05))
                continue

            with rgb_lock:
                frame = rgb_latest_frame
                frame_sequence = rgb_latest_sequence
                frame_capture_wall = rgb_latest_capture_wall

            # Write only if the capture process supplied a genuinely new frame.
            if (
                frame is not None
                and frame_sequence != last_written_sequence
                and frame_capture_wall >= segment_deadline - RGB_SEGMENT_SECONDS
            ):
                if writer is None:
                    wanted = segment_wall.strftime("%Y%m%d_%H%M%S")
                    stem = unique_stem(ar_dir, wanted)
                    path = os.path.join(ar_dir, stem + ".avi")
                    writer = make_writer(
                        path,
                        RGB_SAVE_FPS,
                        (ARDUCAM_WIDTH, ARDUCAM_HEIGHT),
                        is_color=True,
                    )
                    print("Arducam 30-second time window started:", stem)

                writer.write(frame)
                frames_written += 1
                last_written_sequence = frame_sequence

            # Do not burst-write after a delay; burst writing would distort time.
            next_tick += tick_period
            if next_tick < now - tick_period:
                next_tick = now + tick_period

    except Exception as exc:
        print("Arducam writer fatal error:", exc)
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()
            print(f"Arducam final time window saved: {frames_written} real frames")


# ---------------------------------------------------------------------------
# Background image / MAT writers
# ---------------------------------------------------------------------------

def vda_save_worker():
    while True:
        try:
            item = vda_queue.get(timeout=0.5)
        except Empty:
            continue

        try:
            if item is None:
                return

            clip_name, frame_index, rgb_image, depth_mm = item
            clip_dir = os.path.join(BASE_ROOT, "VDA_dataset", clip_name)
            rgb_dir = os.path.join(clip_dir, "rgb")
            depth_dir = os.path.join(clip_dir, "depth")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            rgb_path = os.path.join(rgb_dir, f"{clip_name}_{frame_index:03d}.jpg")
            depth_path = os.path.join(depth_dir, f"{clip_name}_{frame_index:03d}.png")

            rgb_ok = cv2.imwrite(
                rgb_path,
                rgb_image,
                [cv2.IMWRITE_JPEG_QUALITY, VDA_JPEG_QUALITY],
            )
            depth_ok = cv2.imwrite(
                depth_path,
                depth_mm,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            if not rgb_ok or not depth_ok:
                raise RuntimeError("cv2.imwrite returned False")
            if frame_index == VDA_NUM_FRAMES - 1:
                print("VDA clip saved:", clip_name)
        except Exception as exc:
            print("VDA save failed:", exc)
        finally:
            vda_queue.task_done()


def dm_save_worker():
    while True:
        try:
            item = dm_queue.get(timeout=0.5)
        except Empty:
            continue

        try:
            if item is None:
                return

            file_name, xyz_chw = item
            path = os.path.join(BASE_ROOT, "DM", file_name + ".mat")
            scipy.io.savemat(path, mdict={"out": xyz_chw}, oned_as="row")
            print("DM saved:", file_name, xyz_chw.shape)
        except Exception as exc:
            print("DM save failed:", exc)
        finally:
            dm_queue.task_done()


def ir_snapshot_save_worker():
    """Write boundary IR PNG files without delaying emitter activation."""
    while True:
        try:
            item = ir_snapshot_queue.get(timeout=0.5)
        except Empty:
            continue

        try:
            if item is None:
                return

            snapshot_path, infrared_image = item
            saved = cv2.imwrite(
                snapshot_path,
                infrared_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            if not saved:
                raise RuntimeError("cv2.imwrite returned False")
            print("D456 emitter-OFF IR snapshot saved:", snapshot_path)
        except Exception as exc:
            print("D456 emitter-OFF IR snapshot save failed:", exc)
        finally:
            ir_snapshot_queue.task_done()


# ---------------------------------------------------------------------------
# RealSense helpers
# ---------------------------------------------------------------------------

def find_active_video_profile(pipeline_profile, stream_type):
    """Return the active video profile for one requested stream type."""
    for stream_profile in pipeline_profile.get_streams():
        if stream_profile.stream_type() == stream_type:
            return stream_profile.as_video_stream_profile()
    raise RuntimeError(f"Active RealSense stream was not found: {stream_type}")


def serialize_video_profile(stream_name, video_profile):
    """Convert one active RealSense video profile and its intrinsics to JSON."""
    intrinsics = video_profile.get_intrinsics()
    return {
        "stream": stream_name,
        "stream_index": int(video_profile.stream_index()),
        "format": str(video_profile.format()),
        "fps": int(video_profile.fps()),
        "intrinsics": {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "ppx": float(intrinsics.ppx),
            "ppy": float(intrinsics.ppy),
            "distortion_model": str(intrinsics.model),
            "distortion_coefficients": [
                float(value) for value in intrinsics.coeffs
            ],
        },
    }


def serialize_extrinsics(source_name, target_name, source_profile, target_profile):
    """Serialize the SDK transform from source coordinates to target coordinates."""
    extrinsics = source_profile.get_extrinsics_to(target_profile)
    rotation = [float(value) for value in extrinsics.rotation]
    translation = [float(value) for value in extrinsics.translation]
    return {
        "source": source_name,
        "target": target_name,
        "rotation_row_major_3x3": [
            rotation[0:3],
            rotation[3:6],
            rotation[6:9],
        ],
        "translation_meters_xyz": translation,
        "equation": "point_target = R * point_source + t",
    }


def get_optional_device_info(device, option_name):
    """Read a camera-info string when supported by this SDK and device."""
    option_value = getattr(rs.camera_info, option_name, None)
    if option_value is None:
        return None
    try:
        if device.supports(option_value):
            return device.get_info(option_value)
    except Exception:
        pass
    return None


def save_d456_calibration(pipeline_profile, device, depth_scale):
    """Save active Depth, IR1 and RGB calibration plus depth scale."""
    profiles = {
        "depth": find_active_video_profile(pipeline_profile, rs.stream.depth),
        "infrared_1": find_active_video_profile(
            pipeline_profile,
            rs.stream.infrared,
        ),
        "color": find_active_video_profile(pipeline_profile, rs.stream.color),
    }

    device_info = {}
    for option_name in (
        "name",
        "serial_number",
        "firmware_version",
        "recommended_firmware_version",
        "physical_port",
        "usb_type_descriptor",
        "product_id",
        "product_line",
    ):
        value = get_optional_device_info(device, option_name)
        if value is not None:
            device_info[option_name] = value

    serialized_profiles = {
        name: serialize_video_profile(name, video_profile)
        for name, video_profile in profiles.items()
    }

    # Save all six directed transforms so later processing does not have to
    # invert a factory-calibration transform itself.
    serialized_extrinsics = {}
    for source_name, source_profile in profiles.items():
        for target_name, target_profile in profiles.items():
            if source_name == target_name:
                continue
            key = f"{source_name}_to_{target_name}"
            serialized_extrinsics[key] = serialize_extrinsics(
                source_name,
                target_name,
                source_profile,
                target_profile,
            )

    calibration = {
        "schema_version": 1,
        "saved_at_local": dt.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "device": device_info,
        "depth_scale_meters_per_unit": float(depth_scale),
        "active_stream_profiles": serialized_profiles,
        "extrinsics": serialized_extrinsics,
        "coordinate_convention": {
            "translation_unit": "meter",
            "rotation_storage": "row-major 3x3",
            "transform_equation": "point_target = R * point_source + t",
        },
    }

    serial_number = device_info.get("serial_number", "unknown")
    safe_serial = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in serial_number
    )
    calibration_dir = os.path.join(BASE_ROOT, CALIBRATION_DIR_NAME)
    os.makedirs(calibration_dir, exist_ok=True)
    path = os.path.join(calibration_dir, f"D456_{safe_serial}_calibration.json")
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(calibration, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    return path


def set_d456_emitter(depth_sensor, enabled):
    """Force D456 projector state without using alternating-emitter mode."""
    if not depth_sensor.supports(rs.option.emitter_enabled):
        raise RuntimeError("D456 depth sensor does not expose emitter_enabled")

    # Prevent firmware-controlled alternation/auto-off where supported.
    emitter_on_off = getattr(rs.option, "emitter_on_off", None)
    if emitter_on_off is not None and depth_sensor.supports(emitter_on_off):
        try:
            depth_sensor.set_option(emitter_on_off, 0)
        except Exception as exc:
            print("D456 emitter_on_off could not be disabled:", exc)

    emitter_always_on = getattr(rs.option, "emitter_always_on", None)
    if emitter_always_on is not None and depth_sensor.supports(emitter_always_on):
        try:
            depth_sensor.set_option(emitter_always_on, 1 if enabled else 0)
        except Exception as exc:
            print("D456 emitter_always_on could not be changed:", exc)

    depth_sensor.set_option(rs.option.emitter_enabled, 1 if enabled else 0)
    print("D456 emitter", "ON" if enabled else "OFF")


def start_d456():
    pipeline = rs.pipeline()
    # The D435 has no 5 FPS mode; fall back to a 15 FPS stream, which the
    # time-based clip sampler still saves at the configured save FPS.
    for source_fps in (D456_SOURCE_FPS, 15):
        config = rs.config()
        config.enable_stream(
            rs.stream.depth,
            RS_WIDTH,
            RS_HEIGHT,
            rs.format.z16,
            source_fps,
        )
        config.enable_stream(
            rs.stream.infrared,
            1,
            RS_WIDTH,
            RS_HEIGHT,
            rs.format.y8,
            source_fps,
        )
        config.enable_stream(
            rs.stream.color,
            RS_WIDTH,
            RS_HEIGHT,
            rs.format.bgr8,
            source_fps,
        )
        try:
            profile = pipeline.start(config)
            break
        except RuntimeError:
            if source_fps == 15:
                raise

    device = profile.get_device()
    camera_name = device.get_info(rs.camera_info.name)
    upper_name = camera_name.upper()
    # Some librealsense/firmware combinations identify a D456 as D455.
    valid_model = any(
        model in upper_name for model in ("D456", "D455", "D435", "D436")
    )
    if not valid_model:
        pipeline.stop()
        raise RuntimeError(
            "Connected RealSense is not a D456-compatible camera: " + camera_name
        )

    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        calibration_path = save_d456_calibration(profile, device, depth_scale)
        print("D456 calibration saved:", calibration_path)
    except Exception as exc:
        print("D456 calibration save failed; recording will continue:", exc)

    for sensor in device.query_sensors():
        try:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
        except Exception:
            pass

    # Normal idle/prewarm state is emitter OFF. It is enabled only after the
    # scheduled IR snapshot and remains ON during the 20-second videos.
    set_d456_emitter(depth_sensor, False)

    # Runtime capability check: enum availability alone is not enough.  The
    # active frame must actually carry one of the emitter metadata payloads.
    probe_frames = pipeline.wait_for_frames(2000)
    probe_state, probe_source = get_frameset_emitter_state(probe_frames)
    if probe_state is None:
        pipeline.stop()
        raise EmitterMetadataUnavailableError(
            "D456/driver does not expose per-frame emitter metadata "
            "(frame_emitter_mode or frame_laser_power_mode)"
        )
    print(
        "D456 per-frame emitter metadata available:",
        probe_source,
        "state=ON" if probe_state else "state=OFF",
    )

    print("D456 started:", camera_name, "depth scale:", depth_scale)
    return pipeline, depth_sensor, depth_scale


def get_ir_frame(frames):
    try:
        return frames.get_infrared_frame(1)
    except Exception:
        return None


def open_d456_writers(file_name, save_fps):
    paths = {
        "raw_color": os.path.join(BASE_ROOT, "RGB", file_name + ".avi"),
        "depth": os.path.join(BASE_ROOT, "DP", file_name + ".avi"),
        "ir": os.path.join(BASE_ROOT, "IR", file_name + ".avi"),
    }
    writers = {}
    try:
        rs_size = (RS_WIDTH, RS_HEIGHT)
        writers["raw_color"] = make_writer(
            paths["raw_color"], save_fps, rs_size, True
        )
        writers["depth"] = make_writer(
            paths["depth"], save_fps, rs_size, True
        )
        writers["ir"] = make_writer(
            paths["ir"], save_fps, rs_size, False
        )
    except Exception:
        release_writers(writers)
        raise
    return writers


def depth_frame_to_xyz(depth_frame):
    """Fast native SDK point-cloud conversion; replaces the slow Python pixel loop."""
    points = rs.pointcloud().calculate(depth_frame)
    vertices = np.asanyarray(points.get_vertices())
    xyz = vertices.view(np.float32).reshape(-1, 3)
    expected = depth_frame.get_width() * depth_frame.get_height()
    if xyz.shape[0] != expected:
        raise RuntimeError(f"Unexpected point count: {xyz.shape[0]} != {expected}")
    xyz = xyz.reshape(depth_frame.get_height(), depth_frame.get_width(), 3)
    return np.moveaxis(xyz, -1, 0).copy()  # (3, H, W), X/Y/Z in metres


def enqueue_vda_pair(
    frames,
    clip_name,
    frame_index,
    align_to_color,
    spatial_filter,
    temporal_filter,
    depth_scale,
    heartbeat=None,
):
    raw_color_frame = frames.get_color_frame()
    if not raw_color_frame:
        return False

    color_aligned = align_to_color.process(frames)
    depth_to_color = color_aligned.get_depth_frame()
    if not depth_to_color:
        return False

    filtered = spatial_filter.process(depth_to_color)
    filtered = temporal_filter.process(filtered)

    rgb_image = np.asanyarray(raw_color_frame.get_data()).copy()
    depth_raw = np.asanyarray(filtered.get_data()).astype(np.float32)
    depth_mm = np.rint(depth_raw * depth_scale * 1000.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    warned = False
    while not stop_event.is_set():
        touch_heartbeat(heartbeat)
        try:
            vda_queue.put(
                (clip_name, frame_index, rgb_image, depth_mm),
                timeout=0.2,
            )
            return True
        except Full:
            # Do not silently lose a pair now that VDA covers the full clip.
            if not warned:
                print("VDA writer is behind; waiting instead of dropping data")
                warned = True
    return False


def touch_heartbeat(heartbeat):
    if heartbeat is not None:
        heartbeat.value = time.monotonic()


def next_recording_boundary():
    """Return the next Unix wall-clock time divisible by 30 seconds."""
    now = time.time()
    return (int(now // RECORDING_SLOT_SECONDS) + 1) * RECORDING_SLOT_SECONDS


def d456_save_fps_for_slot(scheduled_start_wall):
    slot = dt.datetime.fromtimestamp(scheduled_start_wall)
    if (
        ENABLE_HIGH_RATE_CLIPS
        and slot.minute == 0
        and slot.second == 0
        and slot.hour % HIGH_RATE_INTERVAL_HOURS == 0
    ):
        return D456_HIGH_RATE_SAVE_FPS
    return D456_NORMAL_SAVE_FPS


def should_save_dm_for_slot(scheduled_start_wall):
    """Apply the independent all-clips versus three-hour DM schedule."""
    if SAVE_DM_EVERY_CLIP:
        return True
    slot = dt.datetime.fromtimestamp(scheduled_start_wall)
    return (
        slot.minute == 0
        and slot.second == 0
        and slot.hour % HIGH_RATE_INTERVAL_HOURS == 0
    )


def publish_depth_preview(image, shared_frame, shared_sequence):
    """Publish a BGR depth preview from the D456 child process."""
    if shared_frame is None or shared_sequence is None:
        return
    shared_image = np.frombuffer(shared_frame, dtype=np.uint8).reshape(
        RS_HEIGHT, RS_WIDTH, 3
    )
    sequence = shared_sequence.value
    if sequence % 2:
        sequence += 1
    shared_sequence.value = sequence + 1
    np.copyto(shared_image, image)
    shared_sequence.value = sequence + 2


def publish_live_depth_preview(
    depth_frame,
    shared_frame,
    shared_sequence,
    status_text,
):
    """Publish a live depth frame even when the clip is not being saved."""
    if not depth_frame or shared_frame is None or shared_sequence is None:
        return
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=DEPTH_COLORMAP_ALPHA),
        cv2.COLORMAP_JET,
    )
    depth_color[depth_image == 0] = (0, 0, 0)
    cv2.putText(
        depth_color,
        f"{status_text}  frame={depth_frame.get_frame_number()}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    publish_depth_preview(depth_color, shared_frame, shared_sequence)


def get_frameset_emitter_state(frames):
    """Return (state, source) from per-frame D456 emitter metadata.

    State is True for ON, False for OFF, and None when this librealsense /
    kernel combination does not expose emitter metadata.
    """
    metadata_type = rs.frame_metadata_value
    metadata_names = ("frame_emitter_mode", "frame_laser_power_mode")
    candidate_frames = (frames.get_depth_frame(), get_ir_frame(frames))

    for metadata_name in metadata_names:
        metadata_key = getattr(metadata_type, metadata_name, None)
        if metadata_key is None:
            continue
        for frame in candidate_frames:
            if not frame:
                continue
            try:
                if frame.supports_frame_metadata(metadata_key):
                    value = frame.get_frame_metadata(metadata_key)
                    return value > 0, f"{metadata_name}={value}"
            except Exception:
                continue
    return None, "metadata unavailable"


def wait_for_d456_frames(
    pipeline,
    frame_count,
    heartbeat=None,
    previous_frames=None,
    shared_depth_frame=None,
    shared_depth_sequence=None,
):
    """Return exactly the requested new synchronized frameset."""
    frames = None
    accepted = 0
    stalled_since = None
    previous_streams = {
        "depth": None if previous_frames is None else previous_frames.get_depth_frame(),
        "color": None if previous_frames is None else previous_frames.get_color_frame(),
        "infrared": None if previous_frames is None else get_ir_frame(previous_frames),
    }
    metadata = {}
    for stream_name, frame in previous_streams.items():
        metadata[stream_name] = {
            "number": None if not frame else frame.get_frame_number(),
            "timestamp": None if not frame else frame.get_timestamp(),
        }

    while accepted < frame_count and not stop_event.is_set():
        touch_heartbeat(heartbeat)
        frames = pipeline.wait_for_frames(2000)
        touch_heartbeat(heartbeat)

        current_streams = {
            "depth": frames.get_depth_frame(),
            "color": frames.get_color_frame(),
            "infrared": get_ir_frame(frames),
        }
        publish_live_depth_preview(
            current_streams["depth"],
            shared_depth_frame,
            shared_depth_sequence,
            f"EMITTER ON - SETTLING {accepted + 1}/{frame_count}",
        )
        complete = all(current_streams.values())
        advanced = complete and all(
            metadata[name]["number"] is None
            or (
                frame.get_frame_number() != metadata[name]["number"]
                and frame.get_timestamp() > metadata[name]["timestamp"]
            )
            for name, frame in current_streams.items()
        )
        if advanced:
            for name, frame in current_streams.items():
                metadata[name]["number"] = frame.get_frame_number()
                metadata[name]["timestamp"] = frame.get_timestamp()
            accepted += 1
            stalled_since = None
        elif stalled_since is None:
            stalled_since = time.monotonic()
        elif time.monotonic() - stalled_since >= D456_FRAME_STALL_SECONDS:
            raise RuntimeError(
                "D456 frame number/timestamp stalled after emitter activation"
            )
    return frames if accepted == frame_count else None


def capture_emitter_off_ir_snapshot(
    pipeline,
    depth_sensor,
    file_name,
    scheduled_start_wall,
    heartbeat=None,
    shared_depth_frame=None,
    shared_depth_sequence=None,
):
    """Detect the OFF-to-ON transition and preserve its last OFF IR frame."""
    snapshot_path = os.path.join(BASE_ROOT, "IR_SN", file_name + ".png")
    # The emitter has remained OFF since the preceding clip. Continue
    # consuming real frames throughout idle time and keep the first frameset
    # at or after the scheduled boundary.
    set_d456_emitter(depth_sensor, False)
    snapshot_frames = None
    snapshot_wall = None
    idle_metadata = {
        "depth": {"number": None, "timestamp": None, "stall_since": None},
        "color": {"number": None, "timestamp": None, "stall_since": None},
        "infrared": {"number": None, "timestamp": None, "stall_since": None},
    }
    next_preview = 0.0
    while not stop_event.is_set():
        touch_heartbeat(heartbeat)
        frames = pipeline.wait_for_frames(2000)
        received_wall = time.time()
        received_mono = time.monotonic()
        touch_heartbeat(heartbeat)

        # The camera stays open between recordings, so stale SDK output must
        # also be detected during this emitter-OFF idle interval.  This checks
        # device metadata, never whether the scene/pig happens to move.
        idle_streams = (
            ("depth", frames.get_depth_frame()),
            ("color", frames.get_color_frame()),
            ("infrared", get_ir_frame(frames)),
        )
        for stream_name, frame in idle_streams:
            if not frame:
                raise RuntimeError(
                    f"D456 {stream_name} frame missing while waiting for boundary"
                )
            state = idle_metadata[stream_name]
            frame_number = frame.get_frame_number()
            frame_timestamp = frame.get_timestamp()
            advanced = (
                state["number"] is None
                or (
                    frame_number != state["number"]
                    and frame_timestamp > state["timestamp"]
                )
            )
            if advanced:
                state["number"] = frame_number
                state["timestamp"] = frame_timestamp
                state["stall_since"] = None
            elif state["stall_since"] is None:
                state["stall_since"] = received_mono
                print(f"D456 idle {stream_name} metadata did not advance")
            elif received_mono - state["stall_since"] >= D456_FRAME_STALL_SECONDS:
                raise RuntimeError(
                    f"D456 idle {stream_name} frame number/timestamp stalled"
                )

        if received_mono >= next_preview:
            publish_live_depth_preview(
                idle_streams[0][1],
                shared_depth_frame,
                shared_depth_sequence,
                "IDLE - EMITTER OFF",
            )
            next_preview = received_mono + 1.0 / D456_NORMAL_SAVE_FPS

        if received_wall >= scheduled_start_wall:
            snapshot_frames = frames
            snapshot_wall = received_wall
            break

    boundary_ir = None if snapshot_frames is None else get_ir_frame(snapshot_frames)
    if not boundary_ir:
        raise RuntimeError("D456 IR frame missing while emitter was OFF")
    off_reference_image = np.asanyarray(boundary_ir.get_data()).copy()
    previous_ir_image = off_reference_image
    previous_ir_number = boundary_ir.get_frame_number()
    previous_ir_wall = snapshot_wall
    previous_frames = snapshot_frames

    # Enable the projector, then inspect every new frameset.  Do not write any
    # video frame until its own hardware metadata confirms emitter ON.
    set_d456_emitter(depth_sensor, True)
    detection_deadline = time.monotonic() + EMITTER_DETECTION_TIMEOUT_SECONDS
    video_frames = None
    detection_source = None

    while not stop_event.is_set() and time.monotonic() < detection_deadline:
        candidate = wait_for_d456_frames(
            pipeline,
            1,
            heartbeat,
            previous_frames,
            shared_depth_frame,
            shared_depth_sequence,
        )
        if candidate is None:
            break

        candidate_wall = time.time()
        candidate_ir = get_ir_frame(candidate)
        if not candidate_ir:
            previous_frames = candidate
            continue
        candidate_ir_image = np.asanyarray(candidate_ir.get_data()).copy()

        emitter_state, state_source = get_frameset_emitter_state(candidate)
        if emitter_state is None:
            raise EmitterMetadataUnavailableError(
                "D456 frames do not provide frame_emitter_mode or "
                "frame_laser_power_mode; strict emitter detection cannot run"
            )

        if emitter_state:
            video_frames = candidate
            video_ready_wall = candidate_wall
            detection_source = state_source
            break

        # This frame is still emitter OFF.  If the next frame reports ON, this
        # exact image is the IR snapshot requested for the transition.
        previous_ir_image = candidate_ir_image
        previous_ir_number = candidate_ir.get_frame_number()
        previous_ir_wall = candidate_wall
        previous_frames = candidate

    if video_frames is None:
        raise RuntimeError(
            "D456 emitter ON was not detected within "
            f"{EMITTER_DETECTION_TIMEOUT_SECONDS:.1f} seconds"
        )

    # The immediately preceding frame is known to be emitter OFF. PNG
    # compression and disk I/O happen in a background worker.
    try:
        ir_snapshot_queue.put_nowait((snapshot_path, previous_ir_image))
    except Full as exc:
        raise RuntimeError("D456 IR snapshot save queue is full") from exc
    print(
        "D456 last emitter-OFF IR frame captured:",
        snapshot_path,
        f"boundary delay={previous_ir_wall - scheduled_start_wall:.3f}s",
        f"frame={previous_ir_number}",
    )
    first_video_depth = video_frames.get_depth_frame()
    print(
        "D456 emitter ON detected; first video frame:",
        first_video_depth.get_frame_number() if first_video_depth else "unknown",
        f"via {detection_source}",
    )
    return video_frames, previous_ir_wall, video_ready_wall


def record_one_d456_clip(
    pipeline,
    depth_sensor,
    depth_scale,
    scheduled_start_wall,
    heartbeat=None,
    shared_depth_frame=None,
    shared_depth_sequence=None,
    recording_state=None,
):

    depth_spatial = rs.spatial_filter()
    align_to_color = rs.align(rs.stream.color) if RECORD_VDA else None
    vda_spatial = rs.spatial_filter() if RECORD_VDA else None

    wanted = dt.datetime.fromtimestamp(scheduled_start_wall).strftime("%Y%m%d_%H%M%S")
    file_name = unique_stem(os.path.join(BASE_ROOT, "DP"), wanted)
    save_fps = d456_save_fps_for_slot(scheduled_start_wall)

    writers = open_d456_writers(file_name, save_fps)

    sample_period = 1.0 / save_fps
    frames_written = 0
    vda_index = 0
    first_frame_done = False
    stream_metadata = {
        "depth": {"number": None, "timestamp": None, "stall_since": None},
        "color": {"number": None, "timestamp": None, "stall_since": None},
        "infrared": {"number": None, "timestamp": None, "stall_since": None},
    }
    invalid_data_since = None

    try:
        pending_frames, snapshot_wall, actual_start_wall = capture_emitter_off_ir_snapshot(
            pipeline,
            depth_sensor,
            file_name,
            scheduled_start_wall,
            heartbeat,
            shared_depth_frame,
            shared_depth_sequence,
        )

        # Do not let emitter-OFF history enter either temporal depth filter.
        depth_temporal = rs.temporal_filter()
        vda_temporal = rs.temporal_filter() if RECORD_VDA else None
        record_end_wall = actual_start_wall + D456_RECORD_SECONDS
        next_sample = time.monotonic()

        print(
            "D456 scheduled recording started:",
            file_name,
            f"save FPS={save_fps}",
            f"IR delay={snapshot_wall - scheduled_start_wall:.3f}s",
            f"video delay={actual_start_wall - scheduled_start_wall:.3f}s",
        )
        yellow.off()
        green.on()
        if recording_state is not None:
            recording_state.value = 1

        while not stop_event.is_set() and time.time() < record_end_wall:
            if pending_frames is not None:
                frames = pending_frames
                pending_frames = None
            else:
                touch_heartbeat(heartbeat)
                frames = pipeline.wait_for_frames(2000)

            now = time.monotonic()
            touch_heartbeat(heartbeat)

            # Save a complete 20-second emitter-ON video after the IR snapshot.
            if time.time() >= record_end_wall:
                break

            if now < next_sample:
                continue

            # Move the sampling clock forward without performing a rapid burst
            # of writes when processing was momentarily late.
            while next_sample <= now:
                next_sample += sample_period

            raw_depth = frames.get_depth_frame()
            raw_color = frames.get_color_frame()
            infrared = get_ir_frame(frames)
            if not raw_depth or not raw_color or not infrared:
                print("D456 missing frame; sample skipped")
                if invalid_data_since is None:
                    invalid_data_since = now
                elif now - invalid_data_since >= D456_FRAME_STALL_SECONDS:
                    raise RuntimeError("D456 streams have been incomplete for 2 seconds")
                continue

            # Check each physical stream independently. These SDK metadata
            # values advance even when the animal and all pixels stay unchanged.
            metadata_ok = True
            for stream_name, frame in (
                ("depth", raw_depth),
                ("color", raw_color),
                ("infrared", infrared),
            ):
                state = stream_metadata[stream_name]
                frame_number = frame.get_frame_number()
                frame_timestamp = frame.get_timestamp()
                number_advanced = (
                    state["number"] is None
                    or frame_number != state["number"]
                )
                timestamp_advanced = (
                    state["timestamp"] is None
                    or frame_timestamp > state["timestamp"]
                )

                if not number_advanced or not timestamp_advanced:
                    metadata_ok = False
                    if state["stall_since"] is None:
                        state["stall_since"] = now
                        print(f"D456 {stream_name} metadata did not advance")
                    elif now - state["stall_since"] >= D456_FRAME_STALL_SECONDS:
                        raise RuntimeError(
                            f"D456 {stream_name} frame number/timestamp stalled"
                        )
                else:
                    state["number"] = frame_number
                    state["timestamp"] = frame_timestamp
                    state["stall_since"] = None

            # Do not write any frameset containing stale stream metadata.
            if not metadata_ok:
                continue

            invalid_data_since = None

            processed_depth = depth_spatial.process(raw_depth)
            processed_depth = depth_temporal.process(processed_depth)

            depth_image = np.asanyarray(processed_depth.get_data())
            raw_color_image = np.asanyarray(raw_color.get_data())
            infrared_image = np.asanyarray(infrared.get_data())

            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=DEPTH_COLORMAP_ALPHA),
                cv2.COLORMAP_JET,
            )
            depth_color[depth_image == 0] = (0, 0, 0)

            images = {
                "raw_color": raw_color_image.copy(),
                "depth": depth_color,
                "ir": infrared_image.copy(),
            }
            for key, image in images.items():
                writers[key].write(image)
            frames_written += 1

            publish_depth_preview(
                depth_color,
                shared_depth_frame,
                shared_depth_sequence,
            )

            if not first_frame_done:
                if should_save_dm_for_slot(scheduled_start_wall):
                    try:
                        xyz_chw = depth_frame_to_xyz(raw_depth)
                        dm_queue.put_nowait((file_name, xyz_chw))
                    except Full:
                        print("DM queue full; DM snapshot skipped:", file_name)
                    except Exception as exc:
                        print("DM conversion failed:", exc)
                else:
                    print("DM skipped by three-hour schedule:", file_name)
                first_frame_done = True

            if RECORD_VDA and vda_index < VDA_NUM_FRAMES:
                if enqueue_vda_pair(
                    frames,
                    file_name,
                    vda_index,
                    align_to_color,
                    vda_spatial,
                    vda_temporal,
                    depth_scale,
                    heartbeat,
                ):
                    vda_index += 1

            touch_heartbeat(heartbeat)

        print(
            f"D456 clip saved: {file_name}, "
            f"{frames_written} real frames from a "
            f"{D456_RECORD_SECONDS}-second interval at {save_fps} FPS"
        )
        if RECORD_VDA:
            print(
                f"VDA queued: {file_name}, "
                f"{vda_index}/{VDA_NUM_FRAMES} RGB-depth pairs, "
                f"nominal coverage={vda_index / save_fps:.2f}s"
            )
        else:
            print("VDA recording is disabled")
    finally:
        # End the illuminated interval immediately.  Keep the RealSense
        # pipeline itself running so the next boundary does not pay a USB/
        # librealsense startup penalty.
        try:
            set_d456_emitter(depth_sensor, False)
        except Exception as exc:
            print("D456 could not turn emitter off after recording:", exc)
        release_writers(writers)
        green.off()
        yellow.on()
        if recording_state is not None:
            recording_state.value = 0


def heartbeat_wait(event, seconds, heartbeat):
    deadline = time.monotonic() + seconds
    while not event.is_set():
        touch_heartbeat(heartbeat)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        event.wait(min(remaining, 0.5))


def d456_loop(
    heartbeat=None,
    shared_depth_frame=None,
    shared_depth_sequence=None,
    recording_state=None,
):
    """Keep D456 open and record one strict clip at every :00/:30 slot."""
    pipeline = None
    depth_sensor = None
    depth_scale = None
    scheduled_start_wall = None

    try:
        while not stop_event.is_set():
            # Normal operation starts the pipeline only once.  This branch is
            # entered again only after a genuine camera/SDK failure.
            if pipeline is None:
                try:
                    touch_heartbeat(heartbeat)
                    pipeline, depth_sensor, depth_scale = start_d456()
                    touch_heartbeat(heartbeat)
                    if scheduled_start_wall is None:
                        scheduled_start_wall = next_recording_boundary()
                except EmitterMetadataUnavailableError as exc:
                    print("D456 strict emitter detection disabled recording:", exc)
                    print("D456 will remain idle; restart after fixing metadata support")
                    while not stop_event.is_set():
                        heartbeat_wait(stop_event, 60.0, heartbeat)
                    return
                except Exception as exc:
                    print("D456 start failed; retrying:", exc)
                    pipeline = None
                    depth_sensor = None
                    heartbeat_wait(stop_event, CAMERA_RETRY_SECONDS, heartbeat)
                    continue

            now_wall = time.time()
            while scheduled_start_wall <= now_wall:
                print("D456 skipped a 30-second slot whose boundary already passed")
                scheduled_start_wall += RECORDING_SLOT_SECONDS

            print(
                "D456 next schedule:",
                dt.datetime.fromtimestamp(scheduled_start_wall).strftime("%F %T"),
                "camera remains on; emitter remains off until the boundary",
            )

            try:
                record_one_d456_clip(
                    pipeline,
                    depth_sensor,
                    depth_scale,
                    scheduled_start_wall,
                    heartbeat,
                    shared_depth_frame,
                    shared_depth_sequence,
                    recording_state,
                )
            except EmitterMetadataUnavailableError as exc:
                print("D456 strict emitter detection disabled recording:", exc)
                print("D456 will remain idle; restart after fixing metadata support")
                while not stop_event.is_set():
                    heartbeat_wait(stop_event, 60.0, heartbeat)
                return
            except Exception as exc:
                print("D456 slot failed; camera will be restarted:", exc)
                try:
                    if depth_sensor is not None:
                        set_d456_emitter(depth_sensor, False)
                    pipeline.stop()
                except Exception:
                    pass
                pipeline = None
                depth_sensor = None
                depth_scale = None

            if recording_state is not None:
                recording_state.value = 0
            green.off()
            yellow.on()

            if not stop_event.is_set():
                scheduled_start_wall += RECORDING_SLOT_SECONDS
    finally:
        if recording_state is not None:
            recording_state.value = 0
        if pipeline is not None:
            try:
                if depth_sensor is not None:
                    set_d456_emitter(depth_sensor, False)
                pipeline.stop()
            except Exception:
                pass
        green.off()
        yellow.on()

    print("D456 recorder stopped")


def d456_process_main(
    shared_depth_frame,
    shared_depth_sequence,
    heartbeat,
    recording_state,
    process_stop,
):
    """Own the complete D456 pipeline and its disk workers in one subprocess."""
    global stop_event, vda_queue, dm_queue, ir_snapshot_queue, yellow, green

    stop_event = process_stop
    vda_queue = Queue(maxsize=128)
    dm_queue = Queue(maxsize=4)
    ir_snapshot_queue = Queue(maxsize=8)
    yellow = DummyLED()
    green = DummyLED()

    workers = [
        threading.Thread(target=dm_save_worker, name="dm-writer", daemon=True),
        threading.Thread(
            target=ir_snapshot_save_worker,
            name="ir-snapshot-writer",
            daemon=True,
        ),
    ]
    if RECORD_VDA:
        workers.append(
            threading.Thread(
                target=vda_save_worker,
                name="vda-writer",
                daemon=True,
            )
        )
    for worker in workers:
        worker.start()

    try:
        touch_heartbeat(heartbeat)
        d456_loop(
            heartbeat,
            shared_depth_frame,
            shared_depth_sequence,
            recording_state,
        )
    finally:
        recording_state.value = 0
        process_stop.set()
        if RECORD_VDA:
            vda_queue.put(None)
        dm_queue.put(None)
        ir_snapshot_queue.put(None)
        if RECORD_VDA:
            vda_queue.join()
        dm_queue.join()
        ir_snapshot_queue.join()
        for worker in workers:
            worker.join(timeout=2)


def d456_supervisor_loop(
    process_context,
    shared_depth_frame,
    shared_depth_sequence,
    heartbeat,
    recording_state,
    camera_healthy,
    process_stop,
):
    """Restart the complete D456 process if native SDK code stops responding."""
    global depth_preview

    shared_image = np.frombuffer(shared_depth_frame, dtype=np.uint8).reshape(
        RS_HEIGHT, RS_WIDTH, 3
    )
    process = None
    last_sequence = -1
    last_led_state = None
    process_started_at = 0.0
    startup_heartbeat = 0.0
    process_ready = False
    last_fresh_frame_at = 0.0

    def start_process():
        nonlocal last_sequence, process_started_at, startup_heartbeat, process_ready
        nonlocal last_fresh_frame_at
        camera_healthy.value = 0
        process_stop.clear()
        shared_depth_sequence.value = 0
        heartbeat.value = time.monotonic()
        startup_heartbeat = heartbeat.value
        recording_state.value = 0
        child = process_context.Process(
            target=d456_process_main,
            args=(
                shared_depth_frame,
                shared_depth_sequence,
                heartbeat,
                recording_state,
                process_stop,
            ),
            name="d456-recorder-process",
            daemon=True,
        )
        child.start()
        process_started_at = time.monotonic()
        process_ready = False
        last_sequence = -1
        last_fresh_frame_at = 0.0
        print("D456 subprocess launched, PID:", child.pid)
        return child

    try:
        process = start_process()

        while not stop_event.is_set():
            now = time.monotonic()

            if not process.is_alive():
                camera_healthy.value = 0
                print(
                    "D456 process exited; restarting, exit code:",
                    process.exitcode,
                )
                process.join(timeout=0.2)
                if stop_event.wait(CAMERA_RETRY_SECONDS):
                    break
                process = start_process()
                continue

            if not process_ready and heartbeat.value != startup_heartbeat:
                process_ready = True

            heartbeat_age = now - heartbeat.value
            startup_age = now - process_started_at
            timed_out = (
                process_ready and heartbeat_age >= D456_HARD_BLOCK_SECONDS
            ) or (
                not process_ready
                and startup_age >= PROCESS_STARTUP_TIMEOUT_SECONDS
            )
            if timed_out:
                camera_healthy.value = 0
                old_pid = process.pid
                print(
                    f"D456 process unresponsive for "
                    f"{heartbeat_age if process_ready else startup_age:.1f}s; "
                    f"terminating PID {old_pid}"
                )
                stop_capture_process(process, process_stop)
                if process.is_alive():
                    print("D456 process resisted termination; retrying cleanup")
                    if stop_event.wait(1.0):
                        break
                    continue
                print("D456 blocked process ended; starting a new process")
                process = start_process()
                continue

            sequence_before = shared_depth_sequence.value
            if (
                sequence_before > 0
                and sequence_before % 2 == 0
                and sequence_before != last_sequence
            ):
                candidate = shared_image.copy()
                sequence_after = shared_depth_sequence.value
                if sequence_before == sequence_after and sequence_after % 2 == 0:
                    with preview_lock:
                        depth_preview = candidate
                    last_sequence = sequence_after
                    last_fresh_frame_at = now
                    camera_healthy.value = 1

            if (
                camera_healthy.value
                and now - last_fresh_frame_at >= D456_FRAME_STALL_SECONDS
            ):
                camera_healthy.value = 0
                print("D456 fresh frames stopped; red fault state requested")

            current_led_state = bool(recording_state.value)
            if current_led_state != last_led_state:
                if current_led_state:
                    yellow.off()
                    green.on()
                else:
                    green.off()
                    yellow.on()
                last_led_state = current_led_state

            stop_event.wait(0.02)
    finally:
        camera_healthy.value = 0
        # A normal user shutdown gets enough time to flush VDA/DM queues.
        stop_capture_process(process, process_stop, graceful_timeout=20.0)
        green.off()
        yellow.on()
        print("D456 supervisor stopped")


# ---------------------------------------------------------------------------
# Main / display / shutdown
# ---------------------------------------------------------------------------

def display_loop():
    display_enabled = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if not display_enabled:
        print("No graphical display detected; preview disabled")

    while not stop_event.is_set():
        if display_enabled:
            with preview_lock:
                depth = None if depth_preview is None else depth_preview.copy()
                rgb = None if rgb_preview is None else rgb_preview.copy()

            try:
                if depth is not None:
                    cv2.imshow("RealSense Depth", depth)
                if rgb is not None:
                    cv2.imshow("Arducam", rgb)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
            except cv2.error as exc:
                print("OpenCV display unavailable; recording continues:", exc)
                display_enabled = False

        stop_event.wait(0.01)


def main():
    ensure_directories()
    initialize_leds()

    def request_shutdown(signum, _frame):
        print(f"Received signal {signum}; stopping safely...")
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    # Spawn is intentional. The watchdog may restart capture after other
    # threads already exist; spawning avoids the lock hazards of forking a
    # multi-threaded process.
    process_context = mp.get_context("spawn")

    shared_depth_frame = process_context.Array(
        "B", RS_WIDTH * RS_HEIGHT * 3, lock=False
    )
    shared_depth_sequence = process_context.Value("I", 0, lock=False)
    d456_heartbeat = process_context.Value(
        "d", time.monotonic(), lock=False
    )
    d456_recording_state = process_context.Value("b", 0, lock=False)
    d456_process_stop = process_context.Event()
    d456_camera_healthy = process_context.Value("b", 0, lock=False)

    threads = [
        threading.Thread(
            target=d456_supervisor_loop,
            args=(
                process_context,
                shared_depth_frame,
                shared_depth_sequence,
                d456_heartbeat,
                d456_recording_state,
                d456_camera_healthy,
                d456_process_stop,
            ),
            name="d456-supervisor",
            daemon=True,
        ),
    ]

    if ENABLE_ARDUCAM:
        shared_rgb_frame = process_context.Array(
            "B", ARDUCAM_WIDTH * ARDUCAM_HEIGHT * 3, lock=False
        )
        shared_rgb_sequence = process_context.Value("I", 0, lock=False)
        rgb_capture_heartbeat = process_context.Value(
            "d", time.monotonic(), lock=False
        )
        rgb_process_stop = process_context.Event()
        rgb_camera_healthy = process_context.Value("b", 0, lock=False)
        threads[0:0] = [
            threading.Thread(
                target=arducam_supervisor_loop,
                args=(
                    process_context,
                    shared_rgb_frame,
                    shared_rgb_sequence,
                    rgb_capture_heartbeat,
                    rgb_camera_healthy,
                    rgb_process_stop,
                ),
                name="arducam-supervisor",
                daemon=True,
            ),
            threading.Thread(
                target=arducam_writer_loop,
                name="arducam-writer",
                daemon=True,
            ),
        ]
    else:
        # A disabled camera is not a fault and owns no process/shared buffer.
        rgb_camera_healthy = process_context.Value("b", 1, lock=False)
        print("Arducam is disabled by ENABLE_ARDUCAM=False")

    threads.append(
        threading.Thread(
            target=health_indicator_loop,
            args=(rgb_camera_healthy, d456_camera_healthy),
            name="camera-health-indicator",
            daemon=True,
        )
    )

    for thread in threads:
        thread.start()

    print(
        "Recording started.",
        f"Arducam={'ON' if ENABLE_ARDUCAM else 'OFF'},",
        (
            "DM=every clip"
            if SAVE_DM_EVERY_CLIP
            else "DM=00/03/06/09/12/15/18/21 only"
        ),
    )
    print("Press q or Ctrl+C to stop.")
    try:
        display_loop()
    except KeyboardInterrupt:
        print("Stopping by user request...")
        stop_event.set()
    finally:
        stop_event.set()

        # Supervisors stop their child processes; the D456 child flushes its
        # VDA and DM queues before exiting normally.
        for thread in threads:
            thread.join(timeout=25)

        yellow.off()
        green.off()
        red.off()
        yellow.close()
        green.close()
        red.close()
        cv2.destroyAllWindows()
        print("All recorders stopped safely")


if __name__ == "__main__":
    mp.freeze_support()
    main()



