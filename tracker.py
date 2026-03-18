"""
Cricket AI Tracker — tracker.py  (Enhanced v2)
────────────────────────────────────────────────────────────────────────────
Reads a video source (YouTube Live, RTMP, webcam, file), detects cricket
events using YOLOv8 + optical flow, and pushes REAL suggestions to Firebase.
admin.html shows a confirmation popup for each detected event.

USAGE EXAMPLES:
  # YouTube Live stream
  python tracker.py --source "https://www.youtube.com/watch?v=LIVE_ID"

  # YouTube recorded/highlights
  python tracker.py --source "https://youtu.be/VIDEO_ID"

  # RTMP from Prism / OBS / any encoder
  python tracker.py --source rtmp://localhost:1935/live/stream

  # Local webcam
  python tracker.py --source 0

  # Local video file
  python tracker.py --source match.mp4

  # Run headless (no OpenCV preview window — good for servers)
  python tracker.py --source 0 --no-preview

  # Use better (slower) model for accuracy
  python tracker.py --source match.mp4 --model yolov8s.pt

SETUP:
  pip install -r requirements.txt
  Edit FIREBASE CONFIG section below with your project credentials.

NOTES ON YOUTUBE:
  - yt-dlp is used to resolve the direct stream URL that OpenCV can open.
  - For live streams, yt-dlp fetches the HLS/DASH manifest URL.
  - For VODs, it fetches the best MP4/WebM ≤ 720p.
  - Network quality matters — if frames drop, lower --quality.
  - yt-dlp auto-updates its extractor; keep it fresh:  pip install -U yt-dlp
────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import time
import uuid
import argparse
import sys
import os
from collections import deque
from datetime import datetime

# ── Firebase ────────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ── YOLOv8 ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.\n  Run:  pip install ultralytics")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# FIREBASE CONFIG
# Get from: Firebase Console → Project Settings → Service Accounts
#            → Generate new private key → download JSON
# ════════════════════════════════════════════════════════════════════════════
FIREBASE_CREDENTIALS  = "serviceAccountKey.json"
FIREBASE_DATABASE_URL = "https://cricket-8e338-default-rtdb.firebaseio.com"


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT SOURCE  (overridden by --source CLI argument)
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_SOURCE = "rtmp://localhost:1935/live/stream"


# ════════════════════════════════════════════════════════════════════════════
# DETECTION TUNING
# ════════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD   = 0.45   # YOLO min confidence (0–1)
BALL_CLASS_IDS         = [32]   # COCO 32 = sports ball
PERSON_CLASS_IDS       = [0]    # COCO 0  = person

BOUNDARY_EDGE_FRACTION = 0.08   # ball this close to frame edge → boundary check
SIX_HEIGHT_FRACTION    = 0.25   # ball above this Y fraction from top → six check

WICKET_ZONE_X          = (0.25, 0.75)
WICKET_ZONE_Y          = (0.40, 0.80)

# Cooldowns (seconds) — prevents duplicate triggers on same event
BOUNDARY_COOLDOWN      = 8
WICKET_COOLDOWN        = 10
RUN_COOLDOWN           = 6

TRAJECTORY_HISTORY     = 24    # frames to keep in ball history
FRAME_SKIP             = 2     # process every Nth frame (higher = faster)

# Minimum trajectory frames required before event inference
MIN_TRAJ_FRAMES        = 8

# Camera motion threshold — above this, detections are considered unreliable
CAM_MOTION_THRESH      = 3.5   # pixels/frame


# ════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ════════════════════════════════════════════════════════════════════════════
ball_positions    = deque(maxlen=TRAJECTORY_HISTORY)
person_positions  = deque(maxlen=TRAJECTORY_HISTORY)
prev_gray         = None
camera_motion     = 0.0

last_boundary_time = 0.0
last_wicket_time   = 0.0
last_run_time      = 0.0

firebase_ref = None


# ════════════════════════════════════════════════════════════════════════════
# YOUTUBE / URL RESOLVER
# ════════════════════════════════════════════════════════════════════════════
def resolve_youtube(url: str, quality: str = "720") -> str | None:
    """
    Resolve a YouTube URL to a direct streamable URL.

    For live streams  → returns the HLS manifest URL (OpenCV can open it)
    For VODs          → returns the best MP4/WebM URL at or below `quality`p

    Requires: pip install yt-dlp
    """
    print(f"\n🔗  Resolving YouTube URL …")
    print(f"    {url}")

    try:
        import yt_dlp
    except ImportError:
        print("❌  yt-dlp not installed.\n    Run:  pip install yt-dlp")
        return None

    # Format selector: prefer mp4 at the requested quality, fall back gracefully
    fmt = (
        f"best[height<={quality}][ext=mp4]"
        f"/best[height<={quality}]"
        f"/best[ext=mp4]"
        f"/best"
    )

    ydl_opts = {
        "format":      fmt,
        "quiet":       True,
        "no_warnings": True,
        # For live HLS, yt-dlp returns the manifest — OpenCV ffmpeg backend handles it
        "live_from_start": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if "entries" in info:          # playlist — take first entry
                info = info["entries"][0]

            is_live   = info.get("is_live", False)
            title     = info.get("title",   "Unknown")[:60]
            height    = info.get("height",  "?")
            duration  = info.get("duration", 0)

            # For live streams, yt-dlp surfaces the manifest under 'url'
            direct_url = info.get("url")

            if not direct_url:
                # Walk the formats list manually
                for fmt_entry in reversed(info.get("formats", [])):
                    h = fmt_entry.get("height") or 9999
                    if fmt_entry.get("url") and h <= int(quality):
                        direct_url = fmt_entry["url"]
                        height     = h
                        break

            if not direct_url:
                print("❌  Could not extract a streamable URL from this video.")
                return None

            mins = int(duration // 60) if duration else 0
            secs = int(duration  % 60) if duration else 0
            print(f"✅  Resolved : {title}")
            print(f"    Live     : {'YES — using HLS manifest' if is_live else 'No (VOD)'}")
            print(f"    Quality  : {height}p")
            if duration:
                print(f"    Duration : {mins}m {secs}s")
            print()
            return direct_url

    except Exception as exc:
        err = str(exc)
        if "Private video"   in err: print("❌  Private video — cannot access.")
        elif "age"     in err.lower(): print("❌  Age-restricted — try another video.")
        elif "not available" in err.lower(): print("❌  Video not available in your region.")
        else: print(f"❌  yt-dlp error: {err[:160]}")
        return None


def is_youtube_url(source: str) -> bool:
    return any(x in source for x in ("youtube.com", "youtu.be"))


# ════════════════════════════════════════════════════════════════════════════
# FIREBASE
# ════════════════════════════════════════════════════════════════════════════
def init_firebase() -> bool:
    global firebase_ref
    if not os.path.isfile(FIREBASE_CREDENTIALS):
        print(f"⚠️   Firebase credential file not found: {FIREBASE_CREDENTIALS}")
        print("    Suggestions will be printed to console only.")
        return False
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        firebase_ref = firebase_db.reference("suggestions")
        print("✅  Firebase connected")
        return True
    except Exception as exc:
        print(f"⚠️   Firebase init failed: {exc}")
        return False


def push_suggestion(event_type: str, confidence: float, detail: str = "") -> None:
    """Push a real AI-detected event suggestion to Firebase."""
    sid = str(uuid.uuid4())[:8]
    suggestion = {
        "id":         sid,
        "type":       event_type,       # BOUNDARY | SIX | WICKET | RUN_1 | RUN_2 | RUN_3
        "confidence": round(confidence * 100),
        "detail":     detail,
        "timestamp":  int(time.time() * 1000),
        "status":     "pending",        # admin.html changes to confirmed / dismissed
        "source":     "tracker",        # distinguishes real detections from manual
    }
    ts = datetime.now().strftime("%H:%M:%S")
    pct = suggestion["confidence"]
    print(f"[{ts}]  🤖  DETECTED: {event_type:<10}  {pct:>3}% confidence  — {detail}")

    if firebase_ref:
        try:
            firebase_ref.child(sid).set(suggestion)
        except Exception as exc:
            print(f"      Firebase push failed: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# OPTICAL FLOW — camera motion estimation
# ════════════════════════════════════════════════════════════════════════════
def compute_camera_motion(frame: np.ndarray) -> float:
    """
    Sparse Lucas-Kanade optical flow on background corners to estimate
    camera pan/zoom magnitude (pixels/frame at 320×180 scale).
    High values mean the camera is moving; detections are less reliable.
    """
    global prev_gray, camera_motion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 180))

    if prev_gray is None:
        prev_gray     = gray
        camera_motion = 0.0
        return 0.0

    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=80, qualityLevel=0.01, minDistance=10
    )
    if corners is None or len(corners) < 4:
        prev_gray     = gray
        camera_motion = 0.0
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
    good_prev = corners[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) < 4:
        prev_gray     = gray
        camera_motion = 0.0
        return 0.0

    deltas       = good_next - good_prev
    motion       = float(np.median(np.linalg.norm(deltas, axis=1)))
    prev_gray     = gray
    camera_motion = motion
    return motion


# ════════════════════════════════════════════════════════════════════════════
# TRAJECTORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def analyse_ball_trajectory(frame_h: int, frame_w: int):
    """
    Analyse recent ball positions.
    Returns (event_type, confidence) or (None, 0.0).

    Requires MIN_TRAJ_FRAMES positions so we don't fire on 1-2 noisy detections.
    """
    if len(ball_positions) < MIN_TRAJ_FRAMES:
        return None, 0.0

    positions = list(ball_positions)
    nxs = [p[0] / frame_w for p in positions]
    nys = [p[1] / frame_h for p in positions]

    latest_x, latest_y = nxs[-1], nys[-1]

    # ── SIX: ball reached very high in frame AND is moving fast laterally ──
    window = nys[-min(8, len(nys)):]
    min_y  = min(window)
    if min_y < SIX_HEIGHT_FRACTION:
        dx = abs(nxs[-1] - nxs[max(0, len(nxs) - 8)])
        # Also require consistent upward trajectory (decreasing y)
        upward_frames = sum(
            1 for i in range(1, min(8, len(nys)))
            if nys[-(i+1)] > nys[-i]
        )
        if dx > 0.04 and upward_frames >= 3:
            conf = min(0.92, (SIX_HEIGHT_FRACTION - min_y) * 9.0 + 0.50)
            return "SIX", conf

    # ── BOUNDARY: ball near frame edge AND moving consistently toward it ──
    near_left   = latest_x < BOUNDARY_EDGE_FRACTION
    near_right  = latest_x > (1.0 - BOUNDARY_EDGE_FRACTION)
    near_bottom = latest_y > (1.0 - BOUNDARY_EDGE_FRACTION)

    if near_left or near_right or near_bottom:
        if len(nxs) >= 5:
            dx_trend = nxs[-1] - nxs[-5]
            dy_trend = nys[-1] - nys[-5]
        else:
            dx_trend = dy_trend = 0.0

        moving_toward = (
            (near_left   and dx_trend < -0.015) or
            (near_right  and dx_trend >  0.015) or
            (near_bottom and dy_trend >  0.015)
        )
        if moving_toward:
            # Confidence scales with how close to the edge
            edge_dist = min(latest_x, 1.0 - latest_x, 1.0 - latest_y)
            conf = min(0.88, 0.52 + (BOUNDARY_EDGE_FRACTION - edge_dist) * 6.0)
            return "BOUNDARY", conf

    return None, 0.0


def analyse_runs(frame_h: int, frame_w: int) -> int:
    """
    Detect batsmen crossing by watching two person bboxes swap X-axis positions.
    Returns estimated runs (1–3) or 0.

    This is a rough heuristic — it works best when both batsmen are visible
    and clearly separate in the frame.
    """
    if len(person_positions) < 8:
        return 0

    recent = list(person_positions)[-8:]

    def get_pair(frame_persons):
        if len(frame_persons) >= 2:
            s = sorted(frame_persons, key=lambda p: p[0])
            return s[0][0] / frame_w, s[-1][0] / frame_w
        return None

    first_pair = get_pair(recent[0])  if recent[0]  else None
    last_pair  = get_pair(recent[-1]) if recent[-1] else None

    if first_pair and last_pair:
        # Did the left/right assignment flip?
        swapped = (first_pair[0] > 0.5) != (last_pair[0] > 0.5)
        if swapped:
            return 1
    return 0


def detect_wicket_event(detections: list, frame_h: int, frame_w: int) -> float:
    """
    Wicket heuristic: look for ≥2 small unidentified objects in the wicket zone
    (stumps / bails scattering).  Returns confidence (0.0–1.0).
    """
    small_in_zone = 0
    for x1, y1, x2, y2, conf, cls in detections:
        cx = ((x1 + x2) / 2) / frame_w
        cy = ((y1 + y2) / 2) / frame_h
        w  = (x2 - x1) / frame_w
        h  = (y2 - y1) / frame_h
        area = w * h

        in_zone = (
            WICKET_ZONE_X[0] < cx < WICKET_ZONE_X[1] and
            WICKET_ZONE_Y[0] < cy < WICKET_ZONE_Y[1]
        )
        # Small objects that YOLO isn't confidently classifying as person/ball
        is_small_unknown = area < 0.005 and cls not in BALL_CLASS_IDS + PERSON_CLASS_IDS

        if in_zone and is_small_unknown and conf > 0.30:
            small_in_zone += 1

    if small_in_zone >= 2:
        return min(0.82, 0.50 + small_in_zone * 0.10)
    return 0.0


# ════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ════════════════════════════════════════════════════════════════════════════
def process_stream(source, model, show_preview: bool = True) -> None:
    global last_boundary_time, last_wicket_time, last_run_time

    print(f"\n📡  Opening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌  Cannot open source: {source}")
        print("    For RTMP: make sure your encoder is pushing to that address.")
        print("    For YouTube: re-run resolve_youtube() — stream URLs expire.")
        return

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"✅  Opened: {frame_w}×{frame_h} @ {fps_src:.0f} fps")
    print("    Press Q in preview window to quit.\n")

    frame_count  = 0
    fps_timer    = time.time()
    fps_display  = 0.0
    retry_count  = 0
    MAX_RETRIES  = 5

    while True:
        ret, frame = cap.read()

        if not ret:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                print("❌  Stream repeatedly failed — giving up.")
                break
            wait = min(retry_count * 2, 10)
            print(f"⚠️   Frame read failed (attempt {retry_count}/{MAX_RETRIES}). "
                  f"Retrying in {wait}s …")
            time.sleep(wait)
            cap.release()
            cap = cv2.VideoCapture(source)
            continue

        retry_count = 0
        frame_count += 1

        # FPS display refresh
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps_display = 30 / elapsed if elapsed > 0 else 0
            fps_timer   = time.time()

        # Frame skip for performance
        if frame_count % FRAME_SKIP != 0:
            continue

        now = time.time()

        # ── Camera motion ───────────────────────────────────────────────────
        motion     = compute_camera_motion(frame)
        cam_moving = motion > CAM_MOTION_THRESH

        # ── YOLO detection ──────────────────────────────────────────────────
        results      = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections   = []
        frame_balls  = []
        frame_persons = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf             = float(box.conf[0])
                cls              = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if cls in BALL_CLASS_IDS:
                    frame_balls.append((cx, cy, conf))
                elif cls in PERSON_CLASS_IDS:
                    frame_persons.append((cx, cy, conf))

        # Update position histories
        if frame_balls:
            best = max(frame_balls, key=lambda b: b[2])
            ball_positions.append((best[0], best[1]))

        person_positions.append(frame_persons if frame_persons else [])

        # ── Event detection (skip while camera is panning/zooming) ─────────
        if not cam_moving:

            # — BOUNDARY / SIX ───────────────────────────────────────────────
            if (now - last_boundary_time) > BOUNDARY_COOLDOWN:
                event, conf = analyse_ball_trajectory(frame_h, frame_w)
                if event == "SIX" and conf >= 0.62:
                    push_suggestion(
                        "SIX", conf,
                        f"Ball high trajectory — {len(ball_positions)} pts tracked"
                    )
                    last_boundary_time = now
                    ball_positions.clear()
                elif event == "BOUNDARY" and conf >= 0.56:
                    push_suggestion(
                        "BOUNDARY", conf,
                        f"Ball near frame edge — {len(ball_positions)} pts tracked"
                    )
                    last_boundary_time = now
                    ball_positions.clear()

            # — WICKET ────────────────────────────────────────────────────────
            if (now - last_wicket_time) > WICKET_COOLDOWN and detections:
                wkt_conf = detect_wicket_event(detections, frame_h, frame_w)
                if wkt_conf >= 0.56:
                    push_suggestion(
                        "WICKET", wkt_conf,
                        "Small objects in wicket zone — possible stump/bail disturbance"
                    )
                    last_wicket_time = now

            # — RUNS ──────────────────────────────────────────────────────────
            if (now - last_run_time) > RUN_COOLDOWN:
                runs = analyse_runs(frame_h, frame_w)
                if runs > 0:
                    push_suggestion(
                        f"RUN_{runs}", 0.55,
                        f"Batsmen crossing detected ({runs} run estimated)"
                    )
                    last_run_time = now

        # ── Optional preview window ─────────────────────────────────────────
        if show_preview:
            preview = cv2.resize(frame, (640, 360))
            ph, pw  = preview.shape[:2]
            sf_x    = pw / frame_w
            sf_y    = ph / frame_h

            # Ball markers
            for cx, cy, conf in frame_balls:
                px, py = int(cx * sf_x), int(cy * sf_y)
                cv2.circle(preview, (px, py), 12, (0, 255, 255), 2)
                cv2.putText(preview, f"BALL {conf:.0%}",
                            (px + 14, py), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 255, 255), 1)

            # Person markers
            for cx, cy, _ in frame_persons:
                px, py = int(cx * sf_x), int(cy * sf_y)
                cv2.circle(preview, (px, py), 8, (0, 200, 60), 2)

            # Ball trajectory line
            pts = list(ball_positions)
            for i in range(1, len(pts)):
                p1    = (int(pts[i - 1][0] * sf_x), int(pts[i - 1][1] * sf_y))
                p2    = (int(pts[i][0]     * sf_x), int(pts[i][1]     * sf_y))
                alpha = i / len(pts)
                cv2.line(preview, p1, p2,
                         (0, int(255 * alpha), int(255 * (1 - alpha))), 2)

            # HUD strip
            hud_col = (0, 60, 200) if cam_moving else (0, 200, 80)
            cv2.rectangle(preview, (0, 0), (640, 22), (0, 0, 0), -1)
            cv2.putText(
                preview,
                f"FPS:{fps_display:4.0f}  "
                f"CAM:{motion:.1f}px {'MOVING ' if cam_moving else 'STABLE '}  "
                f"BALL:{len(frame_balls)}  "
                f"PERSONS:{len(frame_persons)}  "
                f"TRAJ:{len(ball_positions)}/{TRAJECTORY_HISTORY}",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, hud_col, 1
            )

            cv2.imshow("Cricket AI Tracker  [Q = quit]", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋  Tracker stopped.")


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Cricket AI Ball Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tracker.py --source "https://www.youtube.com/watch?v=LIVE_ID"
  python tracker.py --source rtmp://localhost:1935/live/stream
  python tracker.py --source 0
  python tracker.py --source match.mp4 --model yolov8s.pt
        """
    )
    parser.add_argument("--source",
                        default=DEFAULT_SOURCE,
                        help="Stream source: YouTube URL, rtmp://…, 0 (webcam), or file path")
    parser.add_argument("--model",
                        default="yolov8n.pt",
                        help="YOLOv8 model (default: yolov8n.pt — fast; yolov8s.pt — better)")
    parser.add_argument("--no-preview",
                        action="store_true",
                        help="Headless mode — no OpenCV preview window")
    parser.add_argument("--conf",
                        type=float,
                        default=CONFIDENCE_THRESHOLD,
                        help=f"YOLO confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--quality",
                        default="720",
                        help="Max video quality for YouTube (default: 720 → 720p)")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.conf

    print("=" * 62)
    print("  🏏  CRICKET AI TRACKER  v2")
    print("=" * 62)
    print(f"  Source   : {args.source}")
    print(f"  Model    : {args.model}")
    print(f"  Preview  : {'OFF (headless)' if args.no_preview else 'ON'}")
    print(f"  Conf     : {CONFIDENCE_THRESHOLD}")
    print(f"  Quality  : {args.quality}p (YouTube only)")
    print("=" * 62)

    # ── Resolve YouTube URLs → direct stream URL ─────────────────────────
    source = args.source
    if is_youtube_url(source):
        source = resolve_youtube(source, quality=args.quality)
        if not source:
            sys.exit(1)

    # ── Firebase ──────────────────────────────────────────────────────────
    fb_ok = init_firebase()
    if not fb_ok:
        print("⚠️   Running without Firebase — events printed to console only.")

    # ── Load YOLO (downloads ~6 MB on first run) ──────────────────────────
    print(f"\n📦  Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"✅  Model ready\n")

    # ── Run ───────────────────────────────────────────────────────────────
    process_stream(
        source       = source,
        model        = model,
        show_preview = not args.no_preview,
    )


if __name__ == "__main__":
    main()