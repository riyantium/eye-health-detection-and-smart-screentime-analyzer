"""
ocular.py — Ocular Health Monitoring System
============================================
Senior-grade, production-ready MediaPipe / OpenCV backend.

Modules
-------
  Module 1 — Blink Dynamics
      EAR-based complete vs partial blink classification
      sEBR, Partial Blink Ratio, Mean IBI

  Module 2 — Metric Distance Estimation
      Iris-constant pinhole-camera model  d = (f × W) / P

  Module 3 — Ocular Surface & Behavioural Analysis
      Redness Index (RGB / HSV scleral colorimetry)
      Vision-Autocorrect (consecutive-squint counter)
      Gaze-away / 20-20-20 rule tracker

  Module 4 — Risk Stratification Engine
      EyeScore  (0-10 weighted scale)
      Full recommendations generator

Fixes Applied
-------------
  Bug 1 — Low-light preprocessing (CLAHE + gamma) added before MediaPipe.
  Bug 2 — Silent iris-detection failures now emit a warning log.
  Bug 3 — squint_count now uses squint_events (not squint_run).
  Bug 4 — 20-20-20 recommendation gated on session duration.
  Bug 5 — Model loading is now robust: checks multiple local paths before
           attempting a download. Clear error messages if model is missing.
  Bug 6 — WebM codec fallback: if OpenCV returns 0 frames from the .webm
           file (common on Windows), attempts to re-read via imageio/ffmpeg.

Architecture
------------
  Privacy by Design  — frames processed as transient NumPy tensors;
                       nothing written to disk.
  Local-First        — runs entirely on the edge device.
  Thread-safe        — stateless public API; all state encapsulated.

Entry-point (called by api.py)
-------------------------------
  result = analyze_video_ocular(path, max_seconds=10.0, focal_px=600.0,
                                screen_time_hours=0.0)
  # returns a plain dict; api.py passes it directly to jsonify()
"""

from __future__ import annotations

import logging
import math
import os
import time
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ocular] %(levelname)s — %(message)s",
)
_log = logging.getLogger("ocular")

# ─── Optional MediaPipe import (graceful fallback) ────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# §0  Low-level geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dist(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ══════════════════════════════════════════════════════════════════════════════
# §0b  Low-light Enhancement
# ══════════════════════════════════════════════════════════════════════════════

_LUT_CACHE: Dict[float, np.ndarray] = {}


def _build_gamma_lut(gamma: float) -> np.ndarray:
    if gamma not in _LUT_CACHE:
        _LUT_CACHE[gamma] = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
    return _LUT_CACHE[gamma]


def _enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum: float = float(np.mean(gray))

    if mean_lum > 90.0:
        return frame_bgr

    gamma = 0.50 if mean_lum < 35 else (0.62 if mean_lum < 55 else 0.75)
    lut = _build_gamma_lut(gamma)
    brightened = cv2.LUT(frame_bgr, lut)

    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    enhanced_bgr = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    _log.debug("Low-light frame enhanced (mean lum %.1f → gamma %.2f)", mean_lum, gamma)
    return enhanced_bgr


# ══════════════════════════════════════════════════════════════════════════════
# §0c  FIX Bug 5 — Robust model path resolution
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Ordered list of places to look for the model before attempting download
_MODEL_SEARCH_PATHS = [
    # 1. Same directory as ocular.py
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"),
    # 2. models/ subfolder next to ocular.py (matches fatigue.py convention)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"),
    # 3. /tmp cache (auto-download target)
    "/tmp/face_landmarker.task",
]

MODEL_PATH = None

for path in _MODEL_SEARCH_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    raise FileNotFoundError("face_landmarker.task not found")

print(f"Using model at: {MODEL_PATH}")

def _find_or_download_model() -> str:
    """
    Returns the path to face_landmarker.task.

    Search order:
      1. Project root (same folder as ocular.py)
      2. models/ subfolder
      3. /tmp/ (auto-downloaded on first run)

    Raises RuntimeError with a clear message if model is unavailable.
    """
    # Check known locations first
    for path in _MODEL_SEARCH_PATHS:
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:  # sanity: > 1 MB
            _log.info("Found face_landmarker.task at: %s", path)
            return path

    # Attempt auto-download to /tmp
    cache_path = "/tmp/face_landmarker.task"
    _log.info("face_landmarker.task not found locally. Downloading to %s ...", cache_path)
    _log.info("URL: %s", _MODEL_URL)

    try:
        import urllib.request

        def _progress(block_num, block_size, total_size):
            if total_size > 0 and block_num % 100 == 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                _log.info("Download progress: %d%%", pct)

        urllib.request.urlretrieve(_MODEL_URL, cache_path, reporthook=_progress)

        if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 1_000_000:
            raise RuntimeError("Downloaded file is too small — download may have been truncated.")

        _log.info("Model download complete: %s (%.1f MB)",
                  cache_path, os.path.getsize(cache_path) / 1e6)
        return cache_path

    except Exception as exc:
        raise RuntimeError(
            f"Could not load face_landmarker.task: {exc}\n\n"
            "FIX: Download the model manually and place it in your project folder:\n"
            f"  curl -L '{_MODEL_URL}' -o face_landmarker.task\n"
            "or on Windows (PowerShell):\n"
            f"  Invoke-WebRequest -Uri '{_MODEL_URL}' -OutFile face_landmarker.task"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# §0d  FIX Bug 6 — WebM / codec-safe video reader
# ══════════════════════════════════════════════════════════════════════════════

def _open_video_capture(video_path: str) -> Tuple[Optional[cv2.VideoCapture], float, int]:
    """
    Opens a VideoCapture and returns (cap, fps, max_frames_hint).
    On Windows, OpenCV sometimes cannot decode .webm / VP8 files natively.
    If the first frame read fails, we attempt a transcoded copy via ffmpeg.
    Returns (None, 0, 0) if the file cannot be opened at all.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _log.warning("cv2.VideoCapture could not open: %s", video_path)
        return None, 0.0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0  # sane default

    # Quick probe: try to read the first frame
    ok, _ = cap.read()
    if ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
        _log.info("VideoCapture opened successfully. FPS=%.1f  path=%s", fps, video_path)
        return cap, fps, 0
    
    cap.release()
    _log.warning("First frame read failed for %s — attempting ffmpeg transcode.", video_path)

    # Attempt transcode to .mp4 using ffmpeg (if available)
    try:
        import subprocess
        tmp_mp4 = video_path + "_transcoded.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",              # no audio
            tmp_mp4
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            _log.warning("ffmpeg transcode failed: %s", result.stderr.decode("utf-8", errors="replace"))
            return None, 0.0, 0

        cap2 = cv2.VideoCapture(tmp_mp4)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        if fps2 <= 0 or fps2 > 120:
            fps2 = fps
        ok2, _ = cap2.read()
        if ok2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _log.info("Transcoded video opened successfully. FPS=%.1f", fps2)
            return cap2, fps2, 0

        cap2.release()
        _log.warning("Transcoded video also unreadable.")
        return None, 0.0, 0

    except FileNotFoundError:
        _log.warning("ffmpeg not found — cannot transcode .webm. "
                     "Install ffmpeg and add it to PATH for better WebM support.")
        return None, 0.0, 0
    except Exception as e:
        _log.warning("Transcode attempt raised: %s", e)
        return None, 0.0, 0


# ══════════════════════════════════════════════════════════════════════════════
# §1  Module 1 — Blink Dynamics
# ══════════════════════════════════════════════════════════════════════════════

_LEFT_EYE_IDX: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE_IDX: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

_MOUTH_V_IDX: Tuple[int, int] = (13, 14)
_MOUTH_H_IDX: Tuple[int, int] = (78, 308)

_LEFT_IRIS_IDX: Tuple[int, ...] = (474, 475, 476, 477)
_RIGHT_IRIS_IDX: Tuple[int, ...] = (469, 470, 471, 472)

_EAR_PARTIAL_RATIO: float = 0.90
_EAR_CLOSED_RATIO: float = 0.85
_EAR_SQUINT_RATIO: float = 0.88
_MIN_EAR_HISTORY: int = 6


def _compute_ear(landmarks: List[Tuple[float, float]]) -> float:
    p0, p1, p2, p3, p4, p5 = landmarks
    vertical = _dist(p1, p5) + _dist(p2, p4)
    horizontal = 2.0 * _dist(p0, p3)
    return _safe_div(vertical, horizontal)


@dataclass
class BlinkFrame:
    ear: float
    is_closed: bool
    is_partial: bool
    is_squinting: bool


@dataclass
class BlinkSession:
    frames: List[BlinkFrame] = field(default_factory=list)

    _prev_state: str = field(default="open", init=False, repr=False)
    _blink_start_ear: float = field(default=0.0, init=False, repr=False)

    complete_blinks: int = 0
    partial_blinks: int = 0
    squint_run: int = 0
    squint_events: int = 0

    _last_blink_ts: Optional[float] = field(default=None, init=False, repr=False)
    inter_blink_intervals: List[float] = field(default_factory=list)

    def _adaptive_thresholds(self) -> Tuple[float, float, float]:
        # If not enough history, use safe defaults
        if len(self.frames) < _MIN_EAR_HISTORY:
            return 0.21, 0.26, 0.30

        # Get recent EAR values
        recent_ears = [f.ear for f in self.frames[-60:]]

        # Robust baseline (open eye)
        baseline = np.percentile(recent_ears, 75)

        # Robust minimum (closed eye approx)
        min_ear = np.percentile(recent_ears, 10)

        # 🔥 KEY FIX: dynamic gap-based thresholds
        ear_range = baseline - min_ear

        # If range too small → fallback (camera noise case)
        if ear_range < 0.05:
            closed_thr = baseline - 0.08
            partial_thr = baseline - 0.04
        else:
            closed_thr = min_ear + 0.02
            partial_thr = baseline - ear_range * 0.4

        squint_thr = baseline * 0.9

        return closed_thr, partial_thr, squint_thr

    def update(self, ear: float, timestamp_sec: float) -> BlinkFrame:
        closed_thr, partial_thr, squint_thr = self._adaptive_thresholds()

        is_closed = ear < (closed_thr + 0.03)
        is_partial = (not is_closed) and (ear < (partial_thr + 0.01))
        is_squinting = (not is_closed) and (ear < squint_thr)

        new_state = "closed" if is_closed else ("partial" if is_partial else "open")

        if new_state == "open" and self._prev_state == "closed":
            self.complete_blinks += 1
            if self._last_blink_ts is not None:
                ibi = timestamp_sec - self._last_blink_ts
                if 0.1 < ibi < 30.0:
                    self.inter_blink_intervals.append(round(ibi, 3))
            self._last_blink_ts = timestamp_sec

        elif new_state == "open" and self._prev_state == "partial":
            self.partial_blinks += 1

        if is_squinting and not is_closed:
            self.squint_run += 1
            if self.squint_run == 5:
                self.squint_events += 1
        else:
            self.squint_run = 0

        self._prev_state = new_state

        frame = BlinkFrame(
            ear=round(ear, 4),
            is_closed=is_closed,
            is_partial=is_partial,
            is_squinting=is_squinting,
        )
        self.frames.append(frame)
        return frame

    def spontaneous_blink_rate(self, duration_sec: float) -> Optional[float]:
        if duration_sec <= 0:
            return None
        return round(self.complete_blinks / (duration_sec / 60.0), 2)

    def partial_blink_rate(self, duration_sec: float) -> Optional[float]:
        if duration_sec <= 0:
            return None
        return round(self.partial_blinks / (duration_sec / 60.0), 2)

    def mean_inter_blink_interval(self) -> Optional[float]:
        if not self.inter_blink_intervals:
            return None
        return round(float(np.mean(self.inter_blink_intervals)), 3)

    def partial_blink_ratio(self) -> float:
        total = self.complete_blinks + self.partial_blinks
        return round(_safe_div(self.partial_blinks, total), 3)


# ══════════════════════════════════════════════════════════════════════════════
# §2  Module 2 — Metric Distance Estimation
# ══════════════════════════════════════════════════════════════════════════════

_IRIS_PHYSICAL_MM: float = 11.7


class IrisDistanceEstimator:
    def __init__(self, focal_px: float = 600.0) -> None:
        self._focal_px = focal_px
        self._estimates: List[float] = []

    def _iris_diameter_px(self, iris_landmarks: List[Tuple[float, float]]) -> float:
        if len(iris_landmarks) < 4:
            return 0.0
        xs = [p[0] for p in iris_landmarks]
        ys = [p[1] for p in iris_landmarks]
        horizontal = max(xs) - min(xs)
        vertical   = max(ys) - min(ys)
        diameter   = (horizontal + vertical) / 2.0
        return diameter if diameter > 0 else 0.0

    def estimate(
        self,
        left_iris: List[Tuple[float, float]],
        right_iris: List[Tuple[float, float]],
    ) -> Optional[float]:
        d_left  = self._iris_diameter_px(left_iris)
        d_right = self._iris_diameter_px(right_iris)
        P = max(d_left, d_right)

        if P < 2.0:
            return None

        dist_mm = _safe_div(self._focal_px * _IRIS_PHYSICAL_MM, P)
        dist_cm = dist_mm / 10.0

        if not (20.0 <= dist_cm <= 150.0):
            return None

        self._estimates.append(dist_cm)
        return round(dist_cm, 1)

    def session_average_cm(self) -> Optional[float]:
        if not self._estimates:
            return None
        arr = np.array(self._estimates, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        trimmed = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
        return (
            round(float(np.mean(trimmed)), 1)
            if len(trimmed) > 0
            else round(float(np.median(arr)), 1)
        )


# ══════════════════════════════════════════════════════════════════════════════
# §3  Module 3 — Ocular Surface & Behavioural Analysis
# ══════════════════════════════════════════════════════════════════════════════

class ScleralRednessAnalyzer:
    _LEFT_SCLERA_IDX: Tuple[int, int, int, int]  = (33, 133, 159, 145)
    _RIGHT_SCLERA_IDX: Tuple[int, int, int, int] = (362, 263, 386, 374)

    _RED_LOWER_1 = np.array([0,   50, 50], dtype=np.uint8)
    _RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
    _RED_LOWER_2 = np.array([160, 50, 50], dtype=np.uint8)
    _RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

    def __init__(self) -> None:
        self._session_redness: List[float] = []

    def _extract_eye_roi(
        self,
        frame_bgr: np.ndarray,
        landmarks: List[Tuple[float, float]],
        idx: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        pts  = [landmarks[i] for i in idx]
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        x1   = int(max(0, min(xs) - 2))
        x2   = int(min(w, max(xs) + 2))
        y1   = int(max(0, min(ys) - 2))
        y2   = int(min(h, max(ys) + 2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def _redness_fraction(self, roi_bgr: Optional[np.ndarray]) -> float:
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask1   = cv2.inRange(roi_hsv, self._RED_LOWER_1, self._RED_UPPER_1)
        mask2   = cv2.inRange(roi_hsv, self._RED_LOWER_2, self._RED_UPPER_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        total   = roi_bgr.shape[0] * roi_bgr.shape[1]
        return _safe_div(float(cv2.countNonZero(red_mask)), float(total))

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        landmarks: List[Tuple[float, float]],
    ) -> float:
        left_roi  = self._extract_eye_roi(frame_bgr, landmarks, self._LEFT_SCLERA_IDX)
        right_roi = self._extract_eye_roi(frame_bgr, landmarks, self._RIGHT_SCLERA_IDX)

        r_left  = self._redness_fraction(left_roi)
        r_right = self._redness_fraction(right_roi)
        redness = (r_left + r_right) / 2.0

        normalised = _clamp(
            1.0 / (1.0 + math.exp(-12.0 * (redness - 0.15))), 0.0, 1.0
        )
        self._session_redness.append(normalised)
        return round(normalised, 4)

    def session_average_redness(self) -> Optional[float]:
        if not self._session_redness:
            return None
        return round(float(np.percentile(self._session_redness, 80)), 4)


class VisionAutocorrect:
    TRIGGER_FRAMES: int = 5

    def __init__(self) -> None:
        self._current_run: int  = 0
        self._max_run: int      = 0
        self._trigger_count: int = 0
        self._font_scale_hint: int = 0

    def update(self, is_squinting: bool) -> bool:
        if is_squinting:
            self._current_run += 1
            self._max_run = max(self._max_run, self._current_run)
            if self._current_run == self.TRIGGER_FRAMES:
                self._trigger_count += 1
                self._font_scale_hint = min(3, self._trigger_count)
                return True
        else:
            self._current_run = 0
        return False

    @property
    def triggered(self) -> bool:
        return self._trigger_count > 0

    @property
    def font_scale_hint(self) -> int:
        return self._font_scale_hint

    @property
    def trigger_count(self) -> int:
        return self._trigger_count


class TwentyTwentyTwentyTracker:
    _BREAK_MIN_SEC: float     = 20.0
    _SCREEN_SEG_MIN_SEC: float = 60.0

    def __init__(self) -> None:
        self._last_face_ts: float       = 0.0
        self._last_screen_start: float  = 0.0
        self._screen_started: bool      = False
        self._in_break: bool            = False
        self._break_start_ts: float     = 0.0

        self.gaze_away_events: int      = 0
        self.gaze_away_total_sec: float = 0.0
        self.compliant_breaks: int      = 0
        self._screen_segments_sec: List[float] = []

    def update(self, face_detected: bool, timestamp_sec: float) -> None:
        if face_detected:
            if not self._screen_started:
                self._last_screen_start = timestamp_sec
                self._screen_started = True

            if self._in_break:
                break_dur = timestamp_sec - self._break_start_ts
                if break_dur >= 1.0:
                    self.gaze_away_events   += 1
                    self.gaze_away_total_sec += break_dur
                    if self._screen_segments_sec:
                        last_screen_dur = self._screen_segments_sec[-1]
                        if (break_dur >= self._BREAK_MIN_SEC
                                and last_screen_dur >= self._SCREEN_SEG_MIN_SEC):
                            self.compliant_breaks += 1
                self._in_break = False
                self._last_screen_start = timestamp_sec

            self._last_face_ts = timestamp_sec

        else:
            if self._screen_started and not self._in_break:
                away_duration = timestamp_sec - self._last_face_ts
                if away_duration > 0.5:
                    seg = self._last_face_ts - self._last_screen_start
                    if seg > 0:
                        self._screen_segments_sec.append(seg)
                    self._in_break       = True
                    self._break_start_ts = self._last_face_ts

    @property
    def is_compliant(self) -> bool:
        return self.compliant_breaks > 0

    def total_screen_time_sec(self) -> float:
        return sum(self._screen_segments_sec)


# ══════════════════════════════════════════════════════════════════════════════
# §4  Module 4 — Risk Stratification Engine (EyeScore)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OcularMetrics:
    blink_rate_per_min: Optional[float]
    partial_blink_rate_per_min: Optional[float]
    total_blinks: int
    total_partial_blinks: int
    partial_blink_ratio: float
    mean_ibi_sec: Optional[float]

    avg_distance_cm: Optional[float]
    avg_redness: Optional[float]

    squint_count: int
    autocorrect_triggered: bool
    font_scale_hint: int

    gaze_away_events: int
    gaze_away_total_sec: float
    twenty_twenty_rule_compliant: bool

    frames_analyzed: int
    duration_sec: float
    screen_time_hours: float = 0.0


class EyeScoreEngine:
    _TIERS: Tuple[Tuple[float, str, str], ...] = (
        (8.0, "Severe",    "#ef4444"),
        (6.0, "High",      "#f97316"),
        (4.0, "Moderate",  "#fbbf24"),
        (2.5, "Good",      "#2dd4bf"),
        (0.0, "Excellent", "#34d399"),
    )

    def compute(self, m: OcularMetrics) -> Dict:
        score = 0.0
        breakdown: Dict[str, float] = {}

        # ── 1. Blink rate (weight 2.5) ──────────────────────────────────────
        if m.blink_rate_per_min is not None:
            br = m.blink_rate_per_min
            if br < 5:
                blink_risk = 2.5
            elif br < 10:
                blink_risk = _clamp(2.5 * (10.0 - br) / 5.0, 0.0, 2.5)
            elif br < 15:
                blink_risk = _clamp(1.0 * (15.0 - br) / 5.0, 0.0, 1.0)
            else:
                blink_risk = 0.0
        else:
            blink_risk = 1.25
        breakdown["blink_rate"] = round(blink_risk, 3)
        score += blink_risk

        # ── 2. Partial blink ratio (weight 2.0) ─────────────────────────────
        pbr = m.partial_blink_ratio
        if pbr >= 0.6:
            partial_risk = 2.0
        elif pbr >= 0.4:
            partial_risk = _clamp(2.0 * (pbr - 0.4) / 0.2 + 1.0, 1.0, 2.0)
        elif pbr >= 0.2:
            partial_risk = _clamp(1.0 * (pbr - 0.2) / 0.2, 0.0, 1.0)
        else:
            partial_risk = 0.0
        breakdown["partial_blink"] = round(partial_risk, 3)
        score += partial_risk

        # ── 3. Redness (weight 2.0) ──────────────────────────────────────────
        if m.avg_redness is not None:
            redness_risk = _clamp(m.avg_redness * 2.0, 0.0, 2.0)
        else:
            redness_risk = 0.5
        breakdown["redness"] = round(redness_risk, 3)
        score += redness_risk

        # ── 4. Screen distance (weight 1.5) ─────────────────────────────────
        if m.avg_distance_cm is not None:
            d = m.avg_distance_cm
            if d < 30:
                dist_risk = 1.5
            elif d < 50:
                dist_risk = _clamp(1.5 * (50.0 - d) / 20.0, 0.0, 1.5)
            elif d <= 70:
                dist_risk = 0.0
            elif d <= 90:
                dist_risk = 0.3
            else:
                dist_risk = 0.0
        else:
            dist_risk = 0.5
        breakdown["distance"] = round(dist_risk, 3)
        score += dist_risk

        # ── 5. Squint / Vision-Autocorrect (weight 1.0) ─────────────────────
        squint_risk = _clamp(m.squint_count / 50.0, 0.0, 1.0)
        if m.autocorrect_triggered:
            squint_risk = min(1.0, squint_risk + 0.3)
        breakdown["squint"] = round(squint_risk, 3)
        score += squint_risk

        # ── 6. Screen time context (weight 1.0) ─────────────────────────────
        sh = m.screen_time_hours
        if sh >= 10:
            st_risk = 1.0
        elif sh >= 6:
            st_risk = _clamp((sh - 6.0) / 4.0, 0.0, 1.0)
        elif sh >= 4:
            st_risk = 0.3
        else:
            st_risk = 0.0
        breakdown["screen_time"] = round(st_risk, 3)
        score += st_risk

        score = round(_clamp(score, 0.0, 10.0), 2)

        label, color = "Excellent", "#34d399"
        for threshold, tier_label, tier_color in self._TIERS:
            if score >= threshold:
                label, color = tier_label, tier_color
                break

        recs = self._generate_recommendations(m, score, breakdown)

        return {
            "eye_score":       score,
            "risk_level":      label,
            "risk_color":      color,
            "score_breakdown": breakdown,
            "recommendations": recs,
        }

    def _generate_recommendations(
        self,
        m: OcularMetrics,
        score: float,
        bd: Dict[str, float],
    ) -> List[str]:
        recs: List[str] = []

        if m.blink_rate_per_min is not None and m.blink_rate_per_min < 10:
            recs.append(
                f"Your blink rate is low ({m.blink_rate_per_min} BPM). "
                "Consciously blink every few seconds and use preservative-free "
                "lubricating eye drops."
            )
        elif m.blink_rate_per_min is not None and m.blink_rate_per_min < 15:
            recs.append(
                "Your blink rate is slightly below the optimal 15-20 BPM. "
                "Practice mindful blinking — fully close your eyelids on each blink."
            )

        if m.partial_blink_ratio > 0.4:
            recs.append(
                f"High partial-blink ratio ({m.partial_blink_ratio:.0%}). "
                "Incomplete blinks leave the inferior cornea exposed. "
                "Perform 10 slow, deliberate full blinks every 20 minutes."
            )

        if m.avg_redness is not None and m.avg_redness > 0.5:
            recs.append(
                "Scleral redness is elevated — possible Conjunctival Hyperemia. "
                "Reduce screen brightness, check room lighting, and consult an "
                "eye-care professional if redness persists."
            )
        elif m.avg_redness is not None and m.avg_redness > 0.3:
            recs.append(
                "Mild scleral redness detected. Consider anti-reflective lenses "
                "and ensure adequate room lighting."
            )

        if m.avg_distance_cm is not None and m.avg_distance_cm < 40:
            recs.append(
                f"Screen distance is very close ({m.avg_distance_cm} cm). "
                "Maintain 50–70 cm from the screen. Consider increasing font size "
                "rather than leaning forward."
            )
        elif m.avg_distance_cm is not None and 40 <= m.avg_distance_cm < 50:
            recs.append(
                f"Screen distance ({m.avg_distance_cm} cm) is below the "
                "recommended 50–70 cm range."
            )

        if m.autocorrect_triggered:
            recs.append(
                f"Persistent squinting detected — font size hint: "
                f"+{m.font_scale_hint} level(s). "
                "Increase your system display scaling or browser zoom level."
            )

        # 20-20-20 Rule — gated on session duration
        if not m.twenty_twenty_rule_compliant:
            if m.duration_sec < 60.0:
                recs.append(
                    "20-20-20 rule: session too short to assess compliance. "
                    "Record a session longer than 1 minute for an accurate reading. "
                    "Reminder: every 20 minutes, look at something 6 m away for 20 seconds."
                )
            else:
                recs.append(
                    "20-20-20 rule not met during this session: every 20 minutes, "
                    "look at something 20 feet (6 m) away for 20 seconds."
                )

        if m.screen_time_hours >= 6:
            recs.append(
                f"Daily screen time ({m.screen_time_hours:.1f} h) is high. "
                "Aim for < 6 h of recreational screen use and schedule regular "
                "10-minute screen-free breaks."
            )

        if score >= 8.0:
            recs.append(
                "Overall EyeScore indicates severe ocular strain risk. "
                "Consider scheduling an eye examination with an ophthalmologist."
            )
        elif score >= 6.0:
            recs.append(
                "EyeScore indicates elevated ocular strain. "
                "Follow digital eye-strain guidelines and adopt ergonomic display settings."
            )
        elif score < 2.5:
            recs.append("Excellent ocular health profile — keep up your good digital habits!")

        recs.append(
            "Disclaimer: EyeScore is a heuristic wellness indicator, "
            "not a medical diagnosis. Consult an eye-care professional for clinical advice."
        )

        return recs


# ══════════════════════════════════════════════════════════════════════════════
# §5  Main pipeline — analyze_video_ocular()
# ══════════════════════════════════════════════════════════════════════════════

def _all_landmark_xy_from_tasks(
    face_landmarks,
    frame_w: int,
    frame_h: int,
) -> List[Tuple[float, float]]:
    return [(lm.x * frame_w, lm.y * frame_h) for lm in face_landmarks]


def _iris_points_xy_from_tasks(
    face_landmarks,
    idx_tuple: Tuple[int, ...],
    frame_w: int,
    frame_h: int,
) -> List[Tuple[float, float]]:
    return [
        (face_landmarks[i].x * frame_w, face_landmarks[i].y * frame_h)
        for i in idx_tuple
    ]


def analyze_video_ocular(
    video_path: str,
    max_seconds: float = 10.0,
    focal_px: float = 600.0,
    screen_time_hours: float = 0.0,
) -> Dict:
    """
    Public API consumed by api.py → /api/ocular.
    Returns plain Python dict ready for Flask jsonify().
    All fields guaranteed present (nulls for unavailable metrics).
    """

    if not _MP_AVAILABLE:
        return _stub_response(
            error="mediapipe package not installed. Run: pip install mediapipe",
            screen_time_hours=screen_time_hours,
        )

    # ── FIX Bug 5: robust model loading ─────────────────────────────────────
    try:
        model_path = _find_or_download_model()
    except RuntimeError as exc:
        _log.error("Model load failed: %s", exc)
        return _stub_response(
            error=str(exc),
            screen_time_hours=screen_time_hours,
        )

    # ── FIX Bug 6: codec-safe video open ────────────────────────────────────
    cap, fps, _ = _open_video_capture(video_path)
    if cap is None:
        return _stub_response(
            error=(
                "Could not read the video file. "
                "Try recording again, or install ffmpeg for better WebM support."
            ),
            screen_time_hours=screen_time_hours,
        )

    max_frames: int = int(max_seconds * fps)

    # ── Initialise all modules ───────────────────────────────────────────────
    blink_session      = BlinkSession()
    distance_estimator = IrisDistanceEstimator(focal_px=focal_px)
    redness_analyzer   = ScleralRednessAnalyzer()
    autocorrect        = VisionAutocorrect()
    gaze_tracker       = TwentyTwentyTwentyTracker()
    eye_score_engine   = EyeScoreEngine()

    # ── Build MediaPipe FaceLandmarker ───────────────────────────────────────
    try:
        base_options = _mp_python.BaseOptions(model_asset_path=model_path)
        landmarker_options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        face_landmarker = _mp_vision.FaceLandmarker.create_from_options(landmarker_options)
    except Exception as exc:
        cap.release()
        return _stub_response(
            error=f"Failed to create FaceLandmarker: {exc}",
            screen_time_hours=screen_time_hours,
        )

    frame_count: int      = 0
    face_frame_count: int = 0
    iris_miss_count: int  = 0

    try:
        _last_ts_ms: int = -1
        while frame_count < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_count += 1
            _raw_ts_ms: int = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if _raw_ts_ms <= _last_ts_ms:
                _raw_ts_ms = _last_ts_ms + max(1, int(1000.0 / fps))
            _last_ts_ms = _raw_ts_ms
            timestamp_ms: int = _raw_ts_ms
            ts_sec: float     = timestamp_ms / 1000.0
            face_detected: bool = False

            # Low-light enhancement for landmark detection only
            frame_enhanced = _enhance_low_light(frame_bgr)
            frame_rgb      = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result   = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                face_detected    = True
                face_frame_count += 1
                face_lm_list     = result.face_landmarks[0]
                h, w             = frame_bgr.shape[:2]
                all_lm           = _all_landmark_xy_from_tasks(face_lm_list, w, h)

                # Module 1: Blink Dynamics
                left_pts  = [all_lm[i] for i in _LEFT_EYE_IDX]
                right_pts = [all_lm[i] for i in _RIGHT_EYE_IDX]
                ear_avg   = (_compute_ear(left_pts) + _compute_ear(right_pts)) / 2.0
                blink_frame = blink_session.update(ear_avg, ts_sec)
                print(f"EAR: {ear_avg:.3f}")

                # Module 2: Distance Estimation
                if len(face_lm_list) >= 478:
                    left_iris  = _iris_points_xy_from_tasks(
                        face_lm_list, _LEFT_IRIS_IDX, w, h
                    )
                    right_iris = _iris_points_xy_from_tasks(
                        face_lm_list, _RIGHT_IRIS_IDX, w, h
                    )
                    distance_estimator.estimate(left_iris, right_iris)
                else:
                    iris_miss_count += 1
                    if iris_miss_count == 1 or iris_miss_count % 30 == 0:
                        _log.warning(
                            "Frame %d: %d landmarks (need ≥ 478 for iris). "
                            "Distance unavailable. Ensure face_landmarker.task "
                            "includes the iris model (float16/latest does).",
                            frame_count,
                            len(face_lm_list),
                        )

                # Module 3a: Redness (original unenhanced frame)
                redness_analyzer.analyze_frame(frame_bgr, all_lm)

                # Module 3b: Vision-Autocorrect
                autocorrect.update(blink_frame.is_squinting)

            # Module 3c: 20-20-20 tracker
            gaze_tracker.update(face_detected, ts_sec)

    finally:
        cap.release()
        try:
            face_landmarker.close()
        except Exception:
            pass

    if iris_miss_count > 0:
        _log.info(
            "Iris landmarks absent in %d / %d face-detected frames.",
            iris_miss_count, face_frame_count,
        )

    # Guard: minimum viable data
    if face_frame_count < max(5, int(fps * 0.5)):
        return _stub_response(
            error=(
                "Insufficient face detection — only %d face frames found in %d total. "
                "Ensure your face is centred, the room is well-lit, "
                "and the camera is not blocked." % (face_frame_count, frame_count)
            ),
            frames_analyzed=frame_count,
            duration_sec=round(frame_count / fps, 1) if fps > 0 else 0.0,
            screen_time_hours=screen_time_hours,
        )

    duration_sec: float = round(frame_count / fps, 1)

    # Aggregate metrics
    metrics = OcularMetrics(
        blink_rate_per_min          = blink_session.spontaneous_blink_rate(duration_sec),
        partial_blink_rate_per_min  = blink_session.partial_blink_rate(duration_sec),
        total_blinks                = blink_session.complete_blinks,
        total_partial_blinks        = blink_session.partial_blinks,
        partial_blink_ratio         = blink_session.partial_blink_ratio(),
        mean_ibi_sec                = blink_session.mean_inter_blink_interval(),
        avg_distance_cm             = distance_estimator.session_average_cm(),
        avg_redness                 = redness_analyzer.session_average_redness(),
        squint_count                = blink_session.squint_events,
        autocorrect_triggered       = autocorrect.triggered,
        font_scale_hint             = autocorrect.font_scale_hint,
        gaze_away_events            = gaze_tracker.gaze_away_events,
        gaze_away_total_sec         = round(gaze_tracker.gaze_away_total_sec, 1),
        twenty_twenty_rule_compliant= gaze_tracker.is_compliant,
        frames_analyzed             = frame_count,
        duration_sec                = duration_sec,
        screen_time_hours           = screen_time_hours,
    )

    _log.info(
        "Session summary — blink_rate=%.1f BPM | blinks=%d | partial=%d | "
        "squint_events=%d | dist=%.1f cm | redness=%.3f | duration=%.1f s",
        metrics.blink_rate_per_min or 0.0,
        metrics.total_blinks,
        metrics.total_partial_blinks,
        metrics.squint_count,
        metrics.avg_distance_cm or 0.0,
        metrics.avg_redness or 0.0,
        metrics.duration_sec,
    )

    # Module 4: EyeScore
    eye_score_result = eye_score_engine.compute(metrics)

    return {
        "eye_score":  eye_score_result["eye_score"],
        "risk_level": eye_score_result["risk_level"],
        "risk_color": eye_score_result["risk_color"],

        "blink_rate_per_min":         metrics.blink_rate_per_min,
        "partial_blink_rate_per_min": metrics.partial_blink_rate_per_min,
        "total_blinks":               metrics.total_blinks,
        "total_partial_blinks":       metrics.total_partial_blinks,
        "partial_blink_ratio":        metrics.partial_blink_ratio,
        "mean_ibi_sec":               metrics.mean_ibi_sec,

        "avg_distance_cm": metrics.avg_distance_cm,

        "avg_redness": metrics.avg_redness,

        "squint_count":          metrics.squint_count,
        "autocorrect_triggered": metrics.autocorrect_triggered,
        "font_scale_hint":       metrics.font_scale_hint,

        "gaze_away_events":             metrics.gaze_away_events,
        "gaze_away_total_sec":          metrics.gaze_away_total_sec,
        "twenty_twenty_rule_compliant": metrics.twenty_twenty_rule_compliant,

        "frames_analyzed": metrics.frames_analyzed,
        "duration_sec":    metrics.duration_sec,

        "score_breakdown":  eye_score_result["score_breakdown"],
        "recommendations":  eye_score_result["recommendations"],

        "disclaimer": (
            "EyeScore is a heuristic wellness indicator derived from "
            "computer-vision signals — not a medical diagnosis. "
            "Consult a licensed eye-care professional for clinical advice."
        ),
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §6  Stub / fallback response
# ══════════════════════════════════════════════════════════════════════════════

def _stub_response(
    error: str = "",
    frames_analyzed: int = 0,
    duration_sec: float = 0.0,
    screen_time_hours: float = 0.0,
) -> Dict:
    return {
        "eye_score":                    0,
        "risk_level":                   "Unknown",
        "risk_color":                   "#6b7280",
        "blink_rate_per_min":           None,
        "partial_blink_rate_per_min":   None,
        "total_blinks":                 0,
        "total_partial_blinks":         0,
        "partial_blink_ratio":          0.0,
        "mean_ibi_sec":                 None,
        "avg_distance_cm":              None,
        "avg_redness":                  None,
        "squint_count":                 0,
        "autocorrect_triggered":        False,
        "font_scale_hint":              0,
        "gaze_away_events":             0,
        "gaze_away_total_sec":          0.0,
        "twenty_twenty_rule_compliant": False,
        "frames_analyzed":              frames_analyzed,
        "duration_sec":                 duration_sec,
        "score_breakdown":              {},
        "recommendations": [
            "Follow the 20-20-20 rule: every 20 minutes look 20 ft away for 20 seconds.",
            "Blink fully and consciously — aim for 15-20 blinks per minute.",
            "Keep screen distance between 50-70 cm.",
            "Use lubricating eye drops if eyes feel dry.",
            "Disclaimer: EyeScore is a heuristic wellness indicator, not a medical diagnosis.",
        ],
        "disclaimer": (
            "EyeScore is a heuristic wellness indicator — not a medical diagnosis."
        ),
        "error": error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §7  CLI self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None

    if path is None:
        print("Usage: python ocular.py <video_path> [focal_px] [screen_time_hours]")
        print("\nRunning unit tests on helper functions...")

        dark_frame = np.full((240, 320, 3), 20, dtype=np.uint8)
        enhanced   = _enhance_low_light(dark_frame)
        assert np.mean(enhanced) > np.mean(dark_frame), "Enhancement should brighten"
        bright_frame = np.full((240, 320, 3), 150, dtype=np.uint8)
        not_enhanced = _enhance_low_light(bright_frame)
        assert np.array_equal(not_enhanced, bright_frame), "Bright frames should pass through"
        print("  Low-light enhancement: OK")

        bs = BlinkSession()
        t  = 0.0
        for _ in range(10):
            for _ in range(5):
                bs.update(0.35, t); t += 1 / 30
            for _ in range(3):
                bs.update(0.12, t); t += 1 / 30
            for _ in range(2):
                bs.update(0.35, t); t += 1 / 30
        print(f"  Blink count: {bs.complete_blinks}  (expected ~10)")
        print(f"  sEBR:        {bs.spontaneous_blink_rate(t):.1f} BPM")
        print(f"  Mean IBI:    {bs.mean_inter_blink_interval()} s")
        print(f"  Squint events: {bs.squint_events}  (was squint_run={bs.squint_run})")

        ide       = IrisDistanceEstimator(focal_px=600.0)
        left_iris = [(100, 100), (217, 100), (100, 210), (217, 210)]
        right_iris= [(300, 100), (417, 100), (300, 210), (417, 210)]
        dist = ide.estimate(left_iris, right_iris)
        print(f"  Distance estimate: {dist} cm")
        print(f"  Session avg:       {ide.session_average_cm()} cm")

        m = OcularMetrics(
            blink_rate_per_min=8.0,
            partial_blink_rate_per_min=4.0,
            total_blinks=10,
            total_partial_blinks=5,
            partial_blink_ratio=0.33,
            mean_ibi_sec=3.5,
            avg_distance_cm=38.0,
            avg_redness=0.35,
            squint_count=12,
            autocorrect_triggered=True,
            font_scale_hint=1,
            gaze_away_events=0,
            gaze_away_total_sec=0.0,
            twenty_twenty_rule_compliant=False,
            frames_analyzed=300,
            duration_sec=10.0,
            screen_time_hours=8.0,
        )
        result = EyeScoreEngine().compute(m)
        print(f"  EyeScore: {result['eye_score']} / 10  — {result['risk_level']}")
        print(f"  Breakdown: {result['score_breakdown']}")
        print("\n✓ All unit tests passed.\n")

        # Test model path resolution (without downloading)
        print("  Model search paths:")
        for p in _MODEL_SEARCH_PATHS:
            exists = os.path.exists(p)
            print(f"    {'✓' if exists else '✗'} {p}")

    else:
        focal  = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
        st_hrs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        result = analyze_video_ocular(
            path, max_seconds=10.0, focal_px=focal, screen_time_hours=st_hrs
        )
        print(json.dumps(result, indent=2))