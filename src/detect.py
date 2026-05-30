"""
Smart Glass AI - Blind Navigation Detection (Raspberry Pi Ready)
=================================================================
Based on the working Windows version, modified for Raspberry Pi 4.

Changes from Windows version:
  - picamera2 (libcamera) support for CSI camera on Debian Trixie/Bookworm
  - Falls back to OpenCV V4L2 if picamera2 unavailable
  - --no-display flag for headless SSH operation
  - --language flag to skip interactive prompt
  - Fixed FPS counter (measures actual frame rate, not processing time)
  - AudioAlerts called with rate parameter for Pi optimization

Usage:
    python3 src/detect.py --device pi --language en       # auto-detect camera
    python3 src/detect.py --device pi --no-display        # headless via SSH
    python3 src/detect.py --device pi --language ar        # Arabic alerts
    python3 src/detect.py --source test_video.mp4         # test with video file
"""

from pathlib import Path
import time
import cv2
import argparse
from collections import defaultdict, deque
from ultralytics import YOLO

from audio_alerts import AudioAlerts

# Try importing picamera2 (libcamera) for Raspberry Pi CSI camera
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "smart_glass" / "weights" / "best.pt"
CAMERA_INDEX = 0

PREDICT_CONF = 0.25
PREDICT_IMGSZ = 640
FRAME_SKIP = 1

LANGUAGE = "en"

# ═══════════════════════════════════════════════════════════════
#  CLASSES (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "traffic_light_red", "traffic_light_yellow", "traffic_light_green",
    "crosswalk", "door", "person", "car", "stairs", "obstacle",
]

DANGER_LEVELS = {
    "traffic_light_red": "DANGER", "traffic_light_yellow": "WARNING",
    "traffic_light_green": "INFO", "crosswalk": "INFO", "door": "INFO",
    "person": "WARNING", "car": "DANGER", "stairs": "DANGER", "obstacle": "WARNING",
}

FRAMES_NEEDED = {"DANGER": 1, "WARNING": 2, "INFO": 3}

COOLDOWNS = {
    "traffic_light_red": 4, "traffic_light_yellow": 4, "traffic_light_green": 8,
    "crosswalk": 12, "door": 10, "person": 6, "car": 5, "stairs": 8, "obstacle": 6,
}

CLASS_PRIORITY = {
    "traffic_light_red": 8, "traffic_light_yellow": 8, "traffic_light_green": 8,
    "crosswalk": 5, "door": 5, "person": 10, "car": 9, "stairs": 9, "obstacle": 1,
}

THRESHOLDS = {
    "traffic_light_red": 0.30, "traffic_light_yellow": 0.30, "traffic_light_green": 0.30,
    "crosswalk": 0.25, "door": 0.30, "person": 0.15, "car": 0.40, "stairs": 0.20,
    "obstacle": 0.25,
}

HISTORY_LEN = 6
TEMPORAL_BOOST = 0.06

# Pi-optimized settings (used when --device pi)
PI_SETTINGS = {
    "predict_imgsz": 320,
    "frame_skip": 3,
    "capture_width": 416,
    "capture_height": 320,
    "speech_rate": 150,
}

# ═══════════════════════════════════════════════════════════════
#  ALERT MESSAGES (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

ALERT_MESSAGES_EN = {
    "traffic_light_red": {"close": "Stop! Red light!", "medium": "Red light ahead. Wait."},
    "traffic_light_yellow": {"close": "Caution! Yellow light.", "medium": "Yellow light ahead."},
    "traffic_light_green": {"close": "Green light. Safe to cross.", "medium": "Green light ahead."},
    "crosswalk": {"close": "Crosswalk here.", "medium": "Crosswalk ahead."},
    "door": {"close": "Door right in front!", "medium": "Door ahead."},
    "person": {"close": "Person very close!", "medium": "Person ahead."},
    "car": {"close": "Car! Stop!", "medium": "Car ahead. Careful."},
    "stairs": {"close": "Stairs! Stop!", "medium": "Stairs ahead. Watch your step."},
    "obstacle": {"close": "Obstacle ahead!", "medium": "Obstacle ahead."},
}

ALERT_MESSAGES_AR = {
    "traffic_light_red": {"close": "قف! إشارة حمراء!", "medium": "إشارة حمراء أمامك. انتظر."},
    "traffic_light_yellow": {"close": "انتبه! إشارة صفراء.", "medium": "إشارة صفراء أمامك."},
    "traffic_light_green": {"close": "إشارة خضراء. يمكنك العبور.", "medium": "إشارة خضراء أمامك."},
    "crosswalk": {"close": "ممر مشاة هنا.", "medium": "ممر مشاة أمامك."},
    "door": {"close": "باب أمامك مباشرة!", "medium": "باب أمامك."},
    "person": {"close": "شخص قريب جداً!", "medium": "شخص أمامك."},
    "car": {"close": "سيارة! قف!", "medium": "سيارة أمامك. انتبه."},
    "stairs": {"close": "درج! قف!", "medium": "درج أمامك. انتبه لخطواتك."},
    "obstacle": {"close": "عائق أمامك!", "medium": "عائق أمامك."},
}


# ═══════════════════════════════════════════════════════════════
#  LANGUAGE SELECTION
# ═══════════════════════════════════════════════════════════════

def choose_language():
    print("\nChoose language:")
    print("1 - Arabic")
    print("2 - English")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        return "ar"

    if choice == "2":
        return "en"

    print("Invalid choice. Defaulting to English.")
    return "en"


# ═══════════════════════════════════════════════════════════════
#  DETECTION FILTERS (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

def passes_filter(label, x1, y1, x2, y2, fw, fh):
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    aspect = bh / bw
    area = (bw * bh) / (fw * fh)

    m = 0.03

    if (x1 <= fw * m and y1 <= fh * m) or (x2 >= fw * (1 - m) and y1 <= fh * m):
        return False

    if (x1 <= fw * m and y2 >= fh * (1 - m)) or (x2 >= fw * (1 - m) and y2 >= fh * (1 - m)):
        return False

    if label == "door":
        if aspect < 1.0 or area < 0.01 or area > 0.60:
            return False

    if label == "person":
        if aspect < 0.5:
            return False

    if label == "obstacle":
        if aspect > 1.5 and area > 0.02:
            return False

    return True


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL BOOST (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

class History:
    def __init__(self):
        self.frames = deque(maxlen=HISTORY_LEN)

    def update(self, detections):
        d = {}

        for det in detections:
            if det["label"] not in d or det["conf"] > d[det["label"]]:
                d[det["label"]] = det["conf"]

        self.frames.append(d)

    def boost(self, label, conf):
        if not self.frames:
            return conf

        seen = sum(1 for f in self.frames if label in f)
        ratio = seen / len(self.frames)
        mult = 3.0 if label == "person" else 1.0

        if ratio >= 0.5:
            return conf + TEMPORAL_BOOST * 2 * mult * ratio

        if seen >= 1:
            return conf + TEMPORAL_BOOST * 0.5 * mult

        return conf


# ═══════════════════════════════════════════════════════════════
#  NMS (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    u = aa + ab - inter

    if u > 0:
        return inter / u

    return 0.0


def nms(dets):
    if not dets:
        return []

    dets.sort(
        key=lambda d: (CLASS_PRIORITY.get(d["label"], 0), d["conf"]),
        reverse=True
    )

    kept = []

    for d in dets:
        ok = True

        for e in kept:
            if iou(d["box"], e["box"]) >= 0.40:
                if CLASS_PRIORITY.get(d["label"], 0) <= CLASS_PRIORITY.get(e["label"], 0):
                    ok = False
                    break

        if ok:
            kept.append(d)

    return kept


# ═══════════════════════════════════════════════════════════════
#  POSITION & DISTANCE HELPERS (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

def position(xc, fw):
    if LANGUAGE == "ar":
        if xc < fw / 3:
            return "على اليسار"
        if xc < 2 * fw / 3:
            return "في المنتصف"
        return "على اليمين"

    if xc < fw / 3:
        return "left"
    if xc < 2 * fw / 3:
        return "center"
    return "right"


def distance(bh, fh):
    r = bh / fh

    if r > 0.30:
        return "close"

    if r > 0.15:
        return "medium"

    return "far"


def get_alert_msg(label, pos, dist):
    messages = ALERT_MESSAGES_AR if LANGUAGE == "ar" else ALERT_MESSAGES_EN
    msgs = messages.get(label, {})

    if dist == "far":
        return None

    msg = msgs.get(dist, msgs.get("medium", None))

    if msg is None:
        return None

    if "traffic_light" not in label:
        msg = "{} {}".format(msg, pos)

    return msg


def startup_message():
    if LANGUAGE == "ar":
        return "تم تشغيل النظارة الذكية."
    return "Smart Glass activated."


def goodbye_message():
    if LANGUAGE == "ar":
        return "إلى اللقاء."
    return "Goodbye."


# ═══════════════════════════════════════════════════════════════
#  DRAWING (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

def color(label):
    level = DANGER_LEVELS.get(label, "INFO")

    if level == "DANGER":
        return (0, 0, 255)

    if level == "WARNING":
        return (0, 255, 255)

    return (0, 200, 0)


def draw(frame, dets, fps, mode):
    h, w = frame.shape[:2]

    for det in dets:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["conf"]
        c = color(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        text = "{} {:.0%}".format(label, conf)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), c, -1)

        cv2.putText(
            frame,
            text,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if dets:
        best = max(dets, key=lambda d: d["conf"])
        t = "DETECTED: {} ({:.0%})".format(
            best["label"].replace("_", " ").upper(),
            best["conf"]
        )
        cv2.putText(frame, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "SCANNING...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(
        frame,
        "FPS: {:.1f} [{}]".format(fps, mode),
        (w - 180, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


# ═══════════════════════════════════════════════════════════════
#  HARDWARE DETECTION (unchanged from Windows version)
# ═══════════════════════════════════════════════════════════════

def detect_hw():
    try:
        with open("/proc/device-tree/model", "r") as f:
            if "raspberry pi" in f.read().lower():
                return "pi"
    except FileNotFoundError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass

    return "cpu"


# ═══════════════════════════════════════════════════════════════
#  PICAMERA2 WRAPPER — NEW for Raspberry Pi
# ═══════════════════════════════════════════════════════════════

class PiCamera2Wrapper:
    """Wraps picamera2 to behave like cv2.VideoCapture.

    On Raspberry Pi with Debian Trixie / Bookworm, OpenCV's V4L2 backend
    cannot decode frames from the CSI camera. picamera2 (libcamera) is
    the only reliable way to access the camera in Python.

    This wrapper provides the same .read() / .isOpened() / .release()
    interface as cv2.VideoCapture, so the detection loop works unchanged.
    """

    def __init__(self, width=640, height=480, fps=15):
        self.picam2 = Picamera2()
        self._started = False
        self._width = width
        self._height = height
        self._fps = fps

        config = self.picam2.create_preview_configuration(
            main={
                "size": (width, height),
                "format": "BGR888",
            },
            controls={
                "FrameDurationLimits": (int(1_000_000 / fps), int(1_000_000 / fps)),
            },
        )
        self.picam2.configure(config)

    def isOpened(self):
        return True

    def read(self):
        """Return (True, bgr_frame) like cv2.VideoCapture.read()."""
        if not self._started:
            self.picam2.start()
            time.sleep(0.5)  # let the first frame settle
            self._started = True

        try:
            frame = self.picam2.capture_array()
            if frame is not None and frame.size > 0:
                return True, frame
        except Exception as e:
            print("  [PiCamera2] read error:", e)
            return False, None

        return False, None

    def set(self, prop_id, value):
        """Minimal set() for interface compatibility."""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
        elif prop_id == cv2.CAP_PROP_FPS:
            self._fps = int(value)

    def release(self):
        if self._started:
            self.picam2.stop()
        self.picam2.close()


# ═══════════════════════════════════════════════════════════════
#  CAMERA OPENING — NEW: picamera2 first, then V4L2 fallback
# ═══════════════════════════════════════════════════════════════

def open_camera(source, is_pi):
    """Open camera with multiple fallback strategies.

    Order of attempts:
      1. picamera2 (libcamera) — works on Raspberry Pi Debian Trixie/Bookworm
      2. V4L2 with MJPG codec — fallback for older Raspberry Pi OS
      3. Default OpenCV capture
      4. Probe other /dev/videoN devices
    """
    dev_int = int(source) if str(source).isdigit() else None

    # ── Strategy 1: picamera2 (libcamera) ──
    if is_pi and HAS_PICAMERA2 and dev_int is not None:
        try:
            wrapper = PiCamera2Wrapper(width=640, height=480, fps=15)
            ret, test_frame = wrapper.read()
            if ret and test_frame is not None:
                h, w = test_frame.shape[:2]
                print("  Camera opened: picamera2/libcamera ({}x{})".format(w, h))
                return wrapper
            else:
                wrapper.release()
                print("  [Camera] picamera2 opened but no frames.")
        except Exception as e:
            print("  [Camera] picamera2 failed:", e)

    # ── Strategy 2: V4L2 with MJPG codec ──
    if dev_int is not None and is_pi:
        cap = cv2.VideoCapture(dev_int, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret, test = cap.read()
            if ret and test is not None:
                print("  Camera opened: /dev/video{} (V4L2+MJPG)".format(dev_int))
                return cap
            cap.release()

    # ── Strategy 3: Default OpenCV capture ──
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret, test = cap.read()
        if ret and test is not None:
            print("  Camera opened: {} (default+MJPG)".format(source))
            return cap
        cap.release()

    # ── Strategy 4: Probe other /dev/videoN devices ──
    if is_pi and dev_int is not None:
        for dev in [0, 2, 1, 3, 4]:
            if dev == dev_int:
                continue
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                ret, _ = cap.read()
                if ret:
                    print("  Camera opened: /dev/video{} (probed)".format(dev))
                    return cap
                cap.release()

    return None


# ═══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ═══════════════════════════════════════════════════════════════

def run(model, source, conf, device_mode, no_display):
    is_pi = device_mode == "pi"

    imgsz = PI_SETTINGS["predict_imgsz"] if is_pi else PREDICT_IMGSZ
    skip = PI_SETTINGS["frame_skip"] if is_pi else FRAME_SKIP

    print("  Mode: {} | Size: {} | Skip: {}".format(device_mode.upper(), imgsz, skip))

    cap = open_camera(source, is_pi)

    if cap is None or not cap.isOpened():
        print("  ERROR: Cannot open camera on source '{}'.".format(source))
        print("")
        print("  On Raspberry Pi, make sure libcamera is installed:")
        print("    sudo apt install python3-libcamera python3-picamera2")
        print("    pip install picamera2")
        print("")
        print("  Then verify: libcamera-hello --list-cameras")
        return

    # Only set OpenCV properties for cv2.VideoCapture (not PiCamera2Wrapper)
    is_picamera2 = isinstance(cap, PiCamera2Wrapper)

    if not is_picamera2:
        if is_pi:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, PI_SETTINGS["capture_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PI_SETTINGS["capture_height"])
            cap.set(cv2.CAP_PROP_FPS, 15)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Audio — use Pi-optimized speech rate
    audio = AudioAlerts(language=LANGUAGE, rate=PI_SETTINGS["speech_rate"] if is_pi else 160)
    audio.start()
    audio.say(startup_message(), allow_repeat=True)

    history = History()
    seen_counts = defaultdict(int)
    last_alert = {}
    last_dets = []
    frame_num = 0
    fps_buf = deque(maxlen=30)
    t_prev = time.time()

    print("  Running... Press 'q' to quit.\n")

    try:
        while True:
            t0 = t_prev
            ret, frame = cap.read()

            if not ret or frame is None:
                # For picamera2, don't break — just skip and try again
                if is_picamera2:
                    time.sleep(0.01)
                    continue
                break

            t_prev = time.time()
            frame_num += 1
            fh, fw = frame.shape[:2]

            if frame_num % skip == 0:
                raw = []

                results = model.predict(
                    source=frame,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False
                )

                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        c = float(box.conf[0])
                        cid = int(box.cls[0])

                        if cid < 0 or cid >= len(CLASS_NAMES):
                            continue

                        label = CLASS_NAMES[cid]

                        thresh = THRESHOLDS.get(label, 0.20)
                        boosted = history.boost(label, c)

                        if boosted < thresh:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if not passes_filter(label, x1, y1, x2, y2, fw, fh):
                            continue

                        raw.append({
                            "label": label,
                            "conf": c,
                            "box": (x1, y1, x2, y2)
                        })

                dets = nms(raw)
                history.update(dets)

                current = set()

                for det in dets:
                    label = det["label"]
                    current.add(label)
                    seen_counts[label] += 1

                for det in dets:
                    label = det["label"]
                    level = DANGER_LEVELS[label]
                    needed = FRAMES_NEEDED[level]

                    if seen_counts[label] < needed:
                        continue

                    now = time.time()
                    cd = COOLDOWNS.get(label, 5)

                    if now - last_alert.get(label, 0) < cd:
                        continue

                    x1, y1, x2, y2 = det["box"]

                    pos = position((x1 + x2) / 2, fw)
                    dist = distance(y2 - y1, fh)
                    msg = get_alert_msg(label, pos, dist)

                    if msg:
                        last_alert[label] = now
                        print("  [{}] {}".format(level, msg))
                        audio.say(msg)

                for label in CLASS_NAMES:
                    if label not in current:
                        seen_counts[label] = max(0, seen_counts[label] - 1)

                last_dets = dets

            # FPS: measure actual time between frames
            elapsed = t_prev - t0
            if elapsed > 0 and frame_num > 1:
                fps_buf.append(1.0 / elapsed)

            fps = sum(fps_buf) / len(fps_buf) if fps_buf else 0

            if not no_display:
                draw(frame, last_dets, fps, device_mode)
                cv2.imshow("Smart Glass AI", frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n  Stopped.")

    finally:
        audio.say(goodbye_message(), allow_repeat=True)
        time.sleep(0.5)
        audio.stop()

        cap.release()

        if not no_display:
            cv2.destroyAllWindows()

        avg = sum(fps_buf) / len(fps_buf) if fps_buf else 0
        print("  {} frames, {:.1f} avg FPS".format(frame_num, avg))


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    global MODEL_PATH
    global LANGUAGE

    parser = argparse.ArgumentParser(
        description="Smart Glass AI - Blind Navigation Detection"
    )
    parser.add_argument("--model", default=None,
                        help="Path to model weights (.pt or .onnx)")
    parser.add_argument("--source", default=str(CAMERA_INDEX),
                        help="Camera index or video file")
    parser.add_argument("--conf", type=float, default=PREDICT_CONF,
                        help="Confidence threshold")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "gpu", "cpu", "pi"],
                        help="Hardware mode (auto-detects Raspberry Pi)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no GUI window). For SSH use.")
    parser.add_argument("--language", default=None, choices=["en", "ar"],
                        help="Alert language (prompts interactively if not set)")

    args = parser.parse_args()

    # Language
    if args.language:
        LANGUAGE = args.language
    else:
        LANGUAGE = choose_language()

    if args.model:
        MODEL_PATH = Path(args.model)

    if args.device == "auto":
        device_mode = detect_hw()
    else:
        device_mode = args.device

    if not MODEL_PATH.exists():
        print("  Model not found: {}".format(MODEL_PATH))
        return

    print("\n{}".format("=" * 55))
    print("  Smart Glass AI - Blind Navigation")
    print("{}".format("=" * 55))
    print("  Loading: {}".format(MODEL_PATH))

    model = YOLO(str(MODEL_PATH))

    source = int(args.source) if args.source.isdigit() else args.source

    run(model, source, args.conf, device_mode, args.no_display)


if __name__ == "__main__":
    main()
