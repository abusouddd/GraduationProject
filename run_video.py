#!/usr/bin/env python3
"""Run Smart Glass AI on a video file and save the output."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import time
import argparse
from pathlib import Path

import detect
detect.HAS_PICAMERA2 = False


def main():
    parser = argparse.ArgumentParser(description="Smart Glass AI - Video Detection")
    parser.add_argument("--source", required=True, help="Input video file (mp4, avi, etc.)")
    parser.add_argument("--output", default=None, help="Output video file (default: output_detected.mp4)")
    parser.add_argument("--model", default=None, help="Path to model weights")
    parser.add_argument("--language", default="en", choices=["en", "ar"])
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print("ERROR: Video file not found:", args.source)
        return

    model_path = args.model or str(detect.MODEL_PATH)
    if not Path(model_path).exists():
        print("ERROR: Model not found:", model_path)
        return

    output = args.output or "output_detected.mp4"

    print("=" * 50)
    print("  Smart Glass AI - Video Detection")
    print("=" * 50)
    print("  Input:  ", args.source)
    print("  Output: ", output)
    print("  Loading model...")

    detect.LANGUAGE = args.language
    model = detect.YOLO(model_path)
    imgsz = detect.PI_SETTINGS["predict_imgsz"]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", args.source)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    if not writer.isOpened():
        # Try XVID if mp4v fails
        output_xvid = output.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(output_xvid, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        if writer.isOpened():
            output = output_xvid
            print("  Using XVID codec:", output)
        else:
            print("ERROR: Cannot create output video")
            cap.release()
            return

    print("  Resolution:", w, "x", h)
    print("  FPS:", fps)
    print("  Total frames:", total)
    print("  Processing...\n")

    history = detect.History()
    frame_num = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Run detection
        fh, fw = frame.shape[:2]
        raw = []

        results = model.predict(source=frame, conf=detect.PREDICT_CONF, imgsz=imgsz, verbose=False)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                c = float(box.conf[0])
                cid = int(box.cls[0])
                if cid < 0 or cid >= len(detect.CLASS_NAMES):
                    continue
                label = detect.CLASS_NAMES[cid]
                thresh = detect.THRESHOLDS.get(label, 0.20)
                boosted = history.boost(label, c)
                if boosted < thresh:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not detect.passes_filter(label, x1, y1, x2, y2, fw, fh):
                    continue
                raw.append({"label": label, "conf": c, "box": (x1, y1, x2, y2)})

        dets = detect.nms(raw)
        history.update(dets)

        # Draw detections on frame
        if dets:
            for det in dets:
                x1, y1, x2, y2 = det["box"]
                label = det["label"]
                conf = det["conf"]
                clr = detect.color(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                text = "{} {:.0%}".format(label, conf)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), clr, -1)
                cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # FPS counter
        elapsed = time.time() - t0
        current_fps = frame_num / elapsed if elapsed > 0 else 0

        # Info bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "FPS: {:.1f} | Frame: {}/{}".format(current_fps, frame_num, total),
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        writer.write(frame)

        # Progress
        if frame_num % 30 == 0 or frame_num == total:
            pct = frame_num / total * 100 if total > 0 else 0
            print("  [{:.0f}%] Frame {}/{} | {:.1f} FPS".format(pct, frame_num, total, current_fps))

    elapsed = time.time() - t0
    avg = frame_num / elapsed if elapsed > 0 else 0

    cap.release()
    writer.release()

    print("\n  Done!")
    print("  Total frames:", frame_num)
    print("  Time:", "{:.1f}s".format(elapsed))
    print("  Average FPS:", "{:.1f}".format(avg))
    print("  Saved to:", output)


if __name__ == "__main__":
    main()
