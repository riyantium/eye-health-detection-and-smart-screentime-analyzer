'''fatigue.py'''

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import cv2


@dataclass
class FatigueResult:
    fatigue_score: float  # 0-100
    eye_closure_ratio: float  # 0-1 (PERCLOS proxy)
    blink_rate_per_min: Optional[float]
    yawn_ratio: Optional[float]  # 0-1 (proxy)
    frames_analyzed: int
    message: str
    method: str  # "mediapipe" or "opencv_haar"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _dist(p1, p2) -> float:
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return (dx * dx + dy * dy) ** 0.5


def _ear(pts) -> float:
    # Eye Aspect Ratio: (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    # pts: list of 6 (x,y) around the eye
    return _safe_div(_dist(pts[1], pts[5]) + _dist(pts[2], pts[4]), 2.0 * _dist(pts[0], pts[3]))


def _try_mediapipe_fatigue(video_path: str, max_seconds: float) -> Optional[FatigueResult]:
    try:
        import mediapipe as mp
    except Exception:
        return None

    # Newer MediaPipe wheels (notably on Python 3.13) may not expose mp.solutions.
    # Prefer Tasks API FaceLandmarker when solutions is unavailable.
    if not hasattr(mp, "solutions"):
        return _try_mediapipe_tasks_fatigue(video_path, max_seconds=max_seconds)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps)

    # FaceMesh landmark indices (MediaPipe)
    # left eye: 33, 160, 158, 133, 153, 144
    # right eye: 362, 385, 387, 263, 373, 380
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    # mouth open proxy: vertical 13-14, horizontal 78-308
    MOUTH_V = (13, 14)
    MOUTH_H = (78, 308)

    frames = 0
    face_frames = 0
    closed_frames = 0
    blinks = 0
    prev_closed = None
    yawn_frames = 0

    ear_values = []
    mar_values = []

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                prev_closed = None
                continue

            face_frames += 1
            lm = res.multi_face_landmarks[0].landmark

            def xy(i):
                p = lm[i]
                return (p.x * w, p.y * h)

            left = [xy(i) for i in LEFT_EYE]
            right = [xy(i) for i in RIGHT_EYE]
            ear = (_ear(left) + _ear(right)) / 2.0
            ear_values.append(ear)

            mouth_v = _dist(xy(MOUTH_V[0]), xy(MOUTH_V[1]))
            mouth_h = _dist(xy(MOUTH_H[0]), xy(MOUTH_H[1]))
            mar = _safe_div(mouth_v, mouth_h)
            mar_values.append(mar)

            # adaptive threshold using a robust baseline from early frames
            if len(ear_values) < 6:
                prev_closed = None
                continue

            baseline = sorted(ear_values[-30:])  # local baseline window
            base = baseline[int(len(baseline) * 0.7)]  # 70th percentile
            ear_thr = max(0.14, base * 0.72)
            is_closed = ear < ear_thr
            if is_closed:
                closed_frames += 1

            if prev_closed is False and is_closed is True:
                # closing transition (start)
                pass
            if prev_closed is True and is_closed is False:
                blinks += 1

            prev_closed = is_closed

            # yawn proxy: mouth aspect ratio above threshold for that frame range
            # threshold slightly adaptive: 80th percentile + margin
            if len(mar_values) >= 10:
                mar_base = sorted(mar_values[-60:])
                mar_thr = max(0.28, mar_base[int(len(mar_base) * 0.8)] + 0.03)
                if mar > mar_thr:
                    yawn_frames += 1

    finally:
        cap.release()
        mesh.close()

    if face_frames < 10:
        return None

    perclos = _safe_div(closed_frames, face_frames)
    duration_min = _safe_div(frames, fps) / 60.0
    blink_rate = _safe_div(blinks, duration_min) if duration_min > 0 else None
    yawn_ratio = _safe_div(yawn_frames, face_frames)

    # scoring: PERCLOS dominates; yawn adds; blink rate adjusts slightly
    score = 15 + 140 * perclos + 35 * yawn_ratio
    if blink_rate is not None:
        if blink_rate < 8:
            score += 6
        elif blink_rate > 32:
            score += 4
    score = _clamp(score, 0.0, 100.0)

    if score < 30:
        msg = "Low fatigue signal detected."
    elif score < 60:
        msg = "Mild fatigue signal. Consider a short break and hydrate."
    elif score < 80:
        msg = "Moderate fatigue signal. Take a 5–10 minute break and rest your eyes."
    else:
        msg = "High fatigue signal. Consider stopping screen work for a while and getting rest."

    return FatigueResult(
        fatigue_score=round(score, 1),
        eye_closure_ratio=round(perclos, 3),
        blink_rate_per_min=(round(blink_rate, 1) if blink_rate is not None else None),
        yawn_ratio=round(yawn_ratio, 3),
        frames_analyzed=int(frames),
        message=msg,
        method="mediapipe",
    )


def _try_mediapipe_tasks_fatigue(video_path: str, max_seconds: float) -> Optional[FatigueResult]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception:
        return None

    import os

    model_path = "models/face_landmarker.task"
    abs_model_path = os.path.join(os.path.dirname(__file__), model_path)
    if not os.path.exists(abs_model_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps)

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH_V = (13, 14)
    MOUTH_H = (78, 308)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=abs_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    frames = 0
    face_frames = 0
    closed_frames = 0
    blinks = 0
    prev_closed = None
    yawn_frames = 0
    ear_values = []
    mar_values = []

    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((frames / fps) * 1000)
            res = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
            if not res.face_landmarks:
                prev_closed = None
                continue

            face_frames += 1
            lm = res.face_landmarks[0]

            def xy(i):
                p = lm[i]
                return (p.x * w, p.y * h)

            left = [xy(i) for i in LEFT_EYE]
            right = [xy(i) for i in RIGHT_EYE]
            ear = (_ear(left) + _ear(right)) / 2.0
            ear_values.append(ear)

            mouth_v = _dist(xy(MOUTH_V[0]), xy(MOUTH_V[1]))
            mouth_h = _dist(xy(MOUTH_H[0]), xy(MOUTH_H[1]))
            mar = _safe_div(mouth_v, mouth_h)
            mar_values.append(mar)

            if len(ear_values) < 6:
                prev_closed = None
                continue

            baseline = sorted(ear_values[-30:])
            base = baseline[int(len(baseline) * 0.7)]
            ear_thr = max(0.14, base * 0.72)
            is_closed = ear < ear_thr
            if is_closed:
                closed_frames += 1

            if prev_closed is True and is_closed is False:
                blinks += 1
            prev_closed = is_closed

            if len(mar_values) >= 10:
                mar_base = sorted(mar_values[-60:])
                mar_thr = max(0.28, mar_base[int(len(mar_base) * 0.8)] + 0.03)
                if mar > mar_thr:
                    yawn_frames += 1

    finally:
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass

    if face_frames < 10:
        return None

    perclos = _safe_div(closed_frames, face_frames)
    duration_min = _safe_div(frames, fps) / 60.0
    blink_rate = _safe_div(blinks, duration_min) if duration_min > 0 else None
    yawn_ratio = _safe_div(yawn_frames, face_frames)

    score = 15 + 140 * perclos + 35 * yawn_ratio
    if blink_rate is not None:
        if blink_rate < 8:
            score += 6
        elif blink_rate > 32:
            score += 4
    score = _clamp(score, 0.0, 100.0)

    if score < 30:
        msg = "Low fatigue signal detected."
    elif score < 60:
        msg = "Mild fatigue signal. Consider a short break and hydrate."
    elif score < 80:
        msg = "Moderate fatigue signal. Take a 5–10 minute break and rest your eyes."
    else:
        msg = "High fatigue signal. Consider stopping screen work for a while and getting rest."

    return FatigueResult(
        fatigue_score=round(score, 1),
        eye_closure_ratio=round(perclos, 3),
        blink_rate_per_min=(round(blink_rate, 1) if blink_rate is not None else None),
        yawn_ratio=round(yawn_ratio, 3),
        frames_analyzed=int(frames),
        message=msg,
        method="mediapipe_tasks",
    )


def _load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    face = cv2.CascadeClassifier(face_path)
    eye = cv2.CascadeClassifier(eye_path)
    return face, eye


def analyze_video_bytes(video_path: str, max_seconds: float = 6.0) -> FatigueResult:
    """
    Local-only fatigue scoring.
    Prefer MediaPipe FaceMesh (EAR/PERCLOS + yawn proxy). Falls back to OpenCV Haar cascades.
    """
    mp_res = _try_mediapipe_fatigue(video_path, max_seconds=max_seconds)
    if mp_res is not None:
        return mp_res

    face_cascade, eye_cascade = _load_cascades()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return FatigueResult(
            fatigue_score=50.0,
            eye_closure_ratio=0.0,
            blink_rate_per_min=None,
            yawn_ratio=None,
            frames_analyzed=0,
            message="Could not read the video. Try again with a clearer, well-lit clip.",
            method="opencv_haar",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps)

    frames = 0
    face_frames = 0
    closed_frames = 0
    state = "unknown"  # open/closed/unknown
    blinks = 0

    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            state = "unknown"
            continue

        face_frames += 1
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        is_closed = len(eyes) == 0
        if is_closed:
            closed_frames += 1

        # blink heuristic: open -> closed -> open transition
        if state == "open" and is_closed:
            state = "closed"
        elif state == "closed" and (not is_closed):
            blinks += 1
            state = "open"
        elif state == "unknown":
            state = "closed" if is_closed else "open"

    cap.release()

    if face_frames == 0:
        return FatigueResult(
            fatigue_score=50.0,
            eye_closure_ratio=0.0,
            blink_rate_per_min=None,
            yawn_ratio=None,
            frames_analyzed=frames,
            message="No face detected. Ensure your face is centered and the room is well-lit.",
            method="opencv_haar",
        )

    eye_closure_ratio = _safe_div(closed_frames, face_frames)
    duration_min = _safe_div(frames, fps) / 60.0
    blink_rate = None
    if duration_min > 0:
        blink_rate = _safe_div(blinks, duration_min)

    # scoring: closure ratio dominates, blink rate adds a small adjustment
    # closure ratio 0.00 -> 10, 0.25 -> 55, 0.50 -> 85, 0.70+ -> 95
    score = 10 + 120 * eye_closure_ratio
    if blink_rate is not None:
        # very low blink rate can indicate staring / fatigue; very high can indicate strain
        if blink_rate < 8:
            score += 8
        elif blink_rate > 30:
            score += 6
    score = _clamp(score, 0.0, 100.0)

    if score < 30:
        msg = "Low fatigue signal detected."
    elif score < 60:
        msg = "Mild fatigue signal. Consider a short break and hydrate."
    elif score < 80:
        msg = "Moderate fatigue signal. Take a 5–10 minute break and rest your eyes."
    else:
        msg = "High fatigue signal. Consider stopping screen work for a while and getting rest."

    return FatigueResult(
        fatigue_score=round(score, 1),
        eye_closure_ratio=round(eye_closure_ratio, 3),
        blink_rate_per_min=(round(blink_rate, 1) if blink_rate is not None else None),
        yawn_ratio=None,
        frames_analyzed=int(frames),
        message=msg,
        method="opencv_haar",
    )


def result_to_json(r: FatigueResult) -> Dict[str, Any]:
    return {
        "fatigue_score": r.fatigue_score,
        "eye_closure_ratio": r.eye_closure_ratio,
        "blink_rate_per_min": r.blink_rate_per_min,
        "yawn_ratio": r.yawn_ratio,
        "frames_analyzed": r.frames_analyzed,
        "message": r.message,
        "method": r.method,
        "disclaimer": "Demo-only heuristic; not a medical diagnosis.",
    }

