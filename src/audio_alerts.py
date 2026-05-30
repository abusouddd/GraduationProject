# -*- coding: utf-8 -*-
"""
Smart Glass AI - Audio Alerts for Raspberry Pi
================================================
Offline TTS using pyttsx3 (espeak backend).
Supports English and Arabic.

Replaces the Windows version which used edge_tts (needs internet).
"""

import threading
import queue
import subprocess
import time

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class AudioAlerts:
    """Offline TTS alert system for Raspberry Pi.

    Falls back through:
      1. pyttsx3 (espeak) — best quality, supports Arabic
      2. espeak CLI      — raw subprocess, always available after apt install
      3. print fallback  — no audio, just console text

    API is identical to the Windows edge_tts version so detect.py works unchanged.
    """

    def __init__(self, language="en", rate=160):
        self.language = language
        self.rate = rate
        self.q = queue.Queue()
        self.running = False
        self.thread = None
        self.last_text = None
        self.last_time = 0

    # ────────────────────────────────────────────────────────────
    #  TTS backends (called from worker thread only)
    # ────────────────────────────────────────────────────────────

    def _speak_pyttsx3(self, text):
        """Use pyttsx3 with espeak backend."""
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", 1.0)

            voices = engine.getProperty("voices")
            chosen = None

            if self.language == "ar":
                # Arabic espeak voice
                for v in voices:
                    if "ar" in v.id.lower():
                        chosen = v.id
                        break
                # Any voice with 'arabic' in the name
                if not chosen:
                    for v in voices:
                        if "arabic" in v.name.lower():
                            chosen = v.id
                            break
            else:
                # English — prefer en-us
                for v in voices:
                    if "en" in v.id.lower() and "us" in v.id.lower():
                        chosen = v.id
                        break
                if not chosen:
                    for v in voices:
                        if "en" in v.id.lower():
                            chosen = v.id
                            break

            if chosen:
                engine.setProperty("voice", chosen)

            engine.say(text)
            engine.runAndWait()

        except Exception as ex:
            print("[Audio pyttsx3 error]", ex)
            self._speak_espeak_cli(text)
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def _speak_espeak_cli(self, text):
        """Direct espeak subprocess — most reliable on Pi."""
        try:
            voice = "ar" if self.language == "ar" else "en-us"
            speed = str(int(self.rate * 1.0))
            subprocess.run(
                ["espeak-ng", "-v", voice, "-s", speed, "-p", "50", text],
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("[Audio] espeak not installed. Run: sudo apt install espeak")
            print("[Fallback]", text)
        except Exception as ex:
            print("[Audio espeak error]", ex)
            print("[Fallback]", text)

    def _speak(self, text):
        """Try pyttsx3 first, fall back to espeak CLI."""
        if HAS_PYTTSX3:
            self._speak_pyttsx3(text)
        else:
            self._speak_espeak_cli(text)

    # ────────────────────────────────────────────────────────────
    #  Worker thread
    # ────────────────────────────────────────────────────────────

    def _worker(self):
        while self.running:
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            print("[Audio] Speaking:", text)
            self._speak(text)

    # ────────────────────────────────────────────────────────────
    #  Public API (identical to Windows version)
    # ────────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def say(self, text, allow_repeat=False, min_gap=1.0):
        if not text:
            return

        text = str(text).strip()
        if not text:
            return

        now = time.time()

        if not allow_repeat and text == self.last_text and (now - self.last_time) < min_gap:
            return

        self.last_text = text
        self.last_time = now

        # Keep queue short — drop old messages to stay responsive
        while self.q.qsize() > 2:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        try:
            self.q.put_nowait(text)
        except Exception as ex:
            print("[Audio Queue error]", ex)

    def stop(self):
        self.running = False

        try:
            self.q.put_nowait(None)
        except Exception:
            pass
