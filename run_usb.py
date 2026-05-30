#!/usr/bin/env python3
"""Run Smart Glass AI with USB camera + Pi speed settings."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import detect

# Disable picamera2 so it uses USB camera via OpenCV
detect.HAS_PICAMERA2 = False

# Force "pi" mode in run() so it uses 320px + frame skip
# We patch argparse to always return "pi" for --device
_orig_parse = detect.argparse.ArgumentParser.parse_args
def _patched_parse(self):
    args = _orig_parse(self)
    if args.device == "cpu":
        args.device = "pi"
    return args
detect.argparse.ArgumentParser.parse_args = _patched_parse

if __name__ == "__main__":
    detect.main()
