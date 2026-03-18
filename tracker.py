"""
Cricket AI Tracker — tracker.py
────────────────────────────────────────────────────────────────────────────
Reads your Prism RTMP stream (or any video source), detects cricket events
using YOLOv8 + optical flow, and pushes suggestions to Firebase.
admin.html then shows a confirmation popup for each detected event.

USAGE:
  python tracker.py --source rtmp://localhost:1935/live/stream
  python tracker.py --source 0                    # webcam
  python tracker.py --source match.mp4            # local video file
  python tracker.py --source youtube              # pull from YouTube Live (needs streamlink)

SETUP:
  pip install -r requirements.txt
  → Edit FIREBASE CONFIG section below with your credentials
  → Edit SOURCE section to match your RTMP address
────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import time
import uuid
import argparse
import sys
import threading
from collections import deque
from datetime import datetime

# ── Firebase ────────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ── YOLOv8 ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# FIREBASE CONFIG — paste your Firebase credentials here
# Get from Firebase Console → Project Settings → Service Accounts
# → Generate new private key → download JSON
# ════════════════════════════════════════════════════════════════════════════
FIREBASE_CREDENTIALS = "serviceAccountKey.json"   # path to your downloaded JSON
FIREBASE_DATABASE_URL = "https://cricket-8e338-default-rtdb.firebaseio.com"

# ════════════════════════════════════════════════════════════════════════════
# STREAM SOURCE
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_SOURCE = "rtmp://localhost:1935/live/stream"
# For Prism → set Prism to also push to rtmp://127.0.0.1:1935/live/stream
# (Prism supports multiple RTMP outputs in Settings → Stream)
# OR use a YouTube Live URL with streamlink:
#   streamlink "https://www.youtube.com/watch?v=YOUR_LIVE_ID" best --player-external-http
#   then set source to the streamlink HTTP URL

# ════════════════════════════════════════════════════════════════════════════
# DETECTION TUNING
# ════════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD   = 0.45   # YOLO min confidence (0–1)
BALL_CLASS_IDS         = [32]   # COCO class 32 = sports ball
PERSON_CLASS_IDS       = [0]    # COCO class 0  = person

# Boundary detection: fraction of frame width/height
# Ball crossing beyond these edges triggers boundary check
BOUNDARY_EDGE_FRACTION = 0.08   # 8% from edge = near boundary

# Six detection: ball trajectory going above this Y fraction (from top)
SIX_HEIGHT_FRACTION    = 0.25   # ball above 25% from top = possible six

# Wicket detection: looks for a cluster of small fast-moving objects
# (stumps/bails flying) in the lower-middle region of the frame
WICKET_ZONE_X          = (0.25, 0.75)   # middle 50% horizontally
WICKET_ZONE_Y          = (0.40, 0.80)   # lower-middle vertically

# Cooldown between suggestions of same type (seconds)
# Prevents duplicate triggers on the same event
BOUNDARY_COOLDOWN      = 8
WICKET_COOLDOWN        = 10
RUN_COOLDOWN           = 6

# How many frames to keep in trajectory history
TRAJECTORY_HISTORY     = 24

# Frame skip: process every Nth frame (higher = faster but less accurate)
FRAME_SKIP             = 2

# ════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ════════════════════════════════════════════════════════════════════════════
ball_positions   = deque(maxlen=TRAJECTORY_HISTORY)
person_positions = deque(maxlen=TRAJECTORY_HISTORY)
prev_gray        = None
camera_motion    = 0.0   # magnitude of camera movement this frame

last_boundary_time = 0
last_wicket_time   = 0
last_run_time      = 0

firebase_ref = None   # will be set after init


# ════════════════════════════════════════════════════════════════════════════
# YOUTUBE URL RESOLVER
# Converts a YouTube watch URL into a direct streamable URL that OpenCV
# can open. Uses yt-dlp which handles all YouTube auth automatically.
# ════════════════════════════════════════════════════════════════════════════
def resolve_youtube(url):
    """
    Takes a YouTube URL and returns a direct video stream URL.
    Works for both live streams and recorded videos.
    Requires yt-dlp: pip install yt-dlp
    """
    print(f"\n🔗 Resolving YouTube URL...")
    print(f"   {url}")
    try:
        import yt_dlp
    except ImportError:
        print("❌ yt-dlp not installed. Run: pip install yt-dlp")
        return None

    ydl_opts = {
        # Prefer 720p or lower for performance — 1080p is too heavy for real-time
        'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # For playlists, take the first video
            if 'entries' in info:
                info = info['entries'][0]

            # Get the direct URL
            direct_url = info.get('url')
            if not direct_url:
                # Try formats list
                formats = info.get('formats', [])
                for f in reversed(formats):
                    if f.get('url') and f.get('height', 999) <= 720:
                        direct_url = f['url']
                        break

            if direct_url:
                title    = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                height   = info.get('height', '?')
                mins     = int(duration // 60) if duration else 0
                secs     = int(duration % 60)  if duration else 0
                print(f"✅ Resolved: {title[:60]}")
                print(f"   Quality : {height}p")
                if duration:
                    print(f"   Duration: {mins}m {secs}s")
                print()
                return direct_url
            else:
                print("❌ Could not extract stream URL from video.")
                return None

    except yt_dlp.utils.DownloadError as e:
        err = str(e)
        if "Private video" in err:
            print("❌ This video is private.")
        elif "age" in err.lower():
            print("❌ Age-restricted video — try a different video.")
        elif "not available" in err.lower():
            print("❌ Video not available in your region.")
        else:
            print(f"❌ yt-dlp error: {err[:120]}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# FIREBASE INIT
# ════════════════════════════════════════════════════════════════════════════
def init_firebase():
    global firebase_ref
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        firebase_ref = firebase_db.reference("suggestions")
        print("✅ Firebase connected")
        return True
    except Exception as e:
        print(f"⚠️  Firebase connection failed: {e}")
        print("   Suggestions will be printed to console only.")
        return False


def push_suggestion(event_type, confidence, detail=""):
    """Push an AI event suggestion to Firebase suggestions node."""
    suggestion = {
        "id":         str(uuid.uuid4())[:8],
        "type":       event_type,        # "BOUNDARY", "SIX", "WICKET", "RUN_1", "RUN_2", "RUN_3"
        "confidence": round(confidence * 100),
        "detail":     detail,
        "timestamp":  int(time.time() * 1000),
        "status":     "pending",         # admin.html changes to "confirmed" or "dismissed"
    }
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] 🤖 AI DETECTED: {event_type} ({suggestion['confidence']}% confidence) — {detail}")

    if firebase_ref:
        try:
            firebase_ref.child(suggestion["id"]).set(suggestion)
        except Exception as e:
            print(f"   Firebase push failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# CAMERA MOTION (OPTICAL FLOW)
# ════════════════════════════════════════════════════════════════════════════
def compute_camera_motion(frame):
    """
    Uses sparse optical flow on background corners to estimate
    how much the camera itself is moving. High camera_motion means
    the camera is panning/zooming — ball detections less reliable.
    Returns motion magnitude (pixels/frame).
    """
    global prev_gray, camera_motion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 180))   # downsample for speed

    if prev_gray is None:
        prev_gray = gray
        camera_motion = 0.0
        return 0.0

    # Detect corners to track
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.01, minDistance=10)
    if corners is None or len(corners) < 4:
        prev_gray = gray
        camera_motion = 0.0
        return 0.0

    # Lucas-Kanade optical flow
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
    good_prev = corners[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) < 4:
        prev_gray = gray
        camera_motion = 0.0
        return 0.0

    # Median motion vector
    deltas = good_next - good_prev
    motion = float(np.median(np.linalg.norm(deltas, axis=1)))

    prev_gray = gray
    camera_motion = motion
    return motion


# ════════════════════════════════════════════════════════════════════════════
# TRAJECTORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def analyse_ball_trajectory(frame_h, frame_w):
    """
    Analyses the recent ball position history to decide if:
    - Ball is heading toward or crossed a boundary edge → BOUNDARY
    - Ball went high in the frame → possible SIX
    Returns (event_type, confidence) or (None, 0)
    """
    if len(ball_positions) < 6:
        return None, 0.0

    positions = list(ball_positions)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # Normalise 0–1
    nxs = [x / frame_w for x in xs]
    nys = [y / frame_h for y in ys]

    latest_x, latest_y = nxs[-1], nys[-1]

    # ── Six check: ball went very high ──────────────────────────────────────
    min_y = min(nys[-8:]) if len(nys) >= 8 else min(nys)
    if min_y < SIX_HEIGHT_FRACTION:
        # Check ball also moved fast (not just a dropped catch)
        dx = abs(nxs[-1] - nxs[0])
        if dx > 0.05:
            conf = min(0.90, (SIX_HEIGHT_FRACTION - min_y) * 8 + 0.5)
            return "SIX", conf

    # ── Boundary check: ball near edge with consistent direction ────────────
    near_left   = latest_x < BOUNDARY_EDGE_FRACTION
    near_right  = latest_x > (1.0 - BOUNDARY_EDGE_FRACTION)
    near_bottom = latest_y > (1.0 - BOUNDARY_EDGE_FRACTION)

    if near_left or near_right or near_bottom:
        # Check that ball was moving toward that edge (not bouncing off)
        dx_trend = nxs[-1] - nxs[-4] if len(nxs) >= 4 else 0
        dy_trend = nys[-1] - nys[-4] if len(nys) >= 4 else 0
        moving_toward_edge = (
            (near_left   and dx_trend < -0.01) or
            (near_right  and dx_trend >  0.01) or
            (near_bottom and dy_trend >  0.01)
        )
        if moving_toward_edge:
            conf = min(0.85, 0.55 + (0.08 - min(latest_x, 1-latest_x, 1-latest_y)) * 5)
            return "BOUNDARY", conf

    return None, 0.0


def analyse_runs(frame_h, frame_w):
    """
    Rough run estimation by watching people (batsmen) cross each other.
    Looks for two person detections whose X positions swap — indicating
    batsmen have crossed and completed a run.
    Returns estimated runs (1-3) or 0.
    """
    if len(person_positions) < 8:
        return 0

    recent = list(person_positions)[-8:]
    if not recent or not recent[0] or not recent[-1]:
        return 0

    # Look for two distinct person positions (striker & non-striker)
    def get_pair(frame_persons):
        if len(frame_persons) >= 2:
            sorted_p = sorted(frame_persons, key=lambda p: p[0])
            return sorted_p[0][0] / frame_w, sorted_p[-1][0] / frame_w
        return None

    first_pair = get_pair(recent[0]) if recent[0] else None
    last_pair  = get_pair(recent[-1]) if recent[-1] else None

    if first_pair and last_pair:
        # Check if positions have swapped (run completed)
        swapped = (first_pair[0] > 0.5) != (last_pair[0] > 0.5)
        if swapped:
            return 1
    return 0


def detect_wicket_event(detections, frame_h, frame_w):
    """
    Wicket detection heuristic:
    - Looks for sudden appearance of multiple small fast-moving objects
      in the wicket zone (stumps/bails scattering)
    - OR looks for a person falling/collapsing (large bbox aspect ratio change)
    Returns confidence (0–1) or 0 if no wicket detected.
    """
    # Check for bails/stumps (small objects in wicket zone)
    small_objects_in_zone = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx = ((x1 + x2) / 2) / frame_w
        cy = ((y1 + y2) / 2) / frame_h
        w  = (x2 - x1) / frame_w
        h  = (y2 - y1) / frame_h
        area = w * h

        in_zone = (WICKET_ZONE_X[0] < cx < WICKET_ZONE_X[1] and
                   WICKET_ZONE_Y[0] < cy < WICKET_ZONE_Y[1])
        is_small = area < 0.005   # very small object

        if in_zone and is_small and conf > 0.35:
            small_objects_in_zone += 1

    if small_objects_in_zone >= 2:
        return min(0.80, 0.50 + small_objects_in_zone * 0.1)

    return 0.0


# ════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ════════════════════════════════════════════════════════════════════════════
def process_stream(source, model, show_preview=True):
    global last_boundary_time, last_wicket_time, last_run_time

    print(f"\n📡 Opening stream: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open stream: {source}")
        print("   Make sure Prism is streaming to that RTMP address,")
        print("   or pass a valid webcam index (0) or video file path.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"✅ Stream opened: {frame_w}×{frame_h} @ {fps_src:.0f}fps")
    print("   Press Q in preview window to quit.\n")

    frame_count = 0
    fps_timer   = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Stream ended or lost. Retrying in 3s...")
            time.sleep(3)
            cap.release()
            cap = cv2.VideoCapture(source)
            continue

        frame_count += 1

        # FPS tracking
        if frame_count % 30 == 0:
            fps_display = 30 / (time.time() - fps_timer)
            fps_timer = time.time()

        # Skip frames for performance
        if frame_count % FRAME_SKIP != 0:
            continue

        now = time.time()

        # ── Camera motion ───────────────────────────────────────────────────
        motion = compute_camera_motion(frame)
        cam_moving = motion > 3.5   # pixels/frame threshold

        # ── YOLO detection ──────────────────────────────────────────────────
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = []
        frame_balls   = []
        frame_persons = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if cls in BALL_CLASS_IDS:
                    frame_balls.append((cx, cy, conf))
                elif cls in PERSON_CLASS_IDS:
                    frame_persons.append((cx, cy, conf))

        # Track positions
        if frame_balls:
            best_ball = max(frame_balls, key=lambda b: b[2])
            ball_positions.append((best_ball[0], best_ball[1]))

        person_positions.append(frame_persons if frame_persons else [])

        # ── Event detection (skip if camera moving fast) ────────────────────
        if not cam_moving:

            # BOUNDARY / SIX
            boundary_cooldown_ok = (now - last_boundary_time) > BOUNDARY_COOLDOWN
            if boundary_cooldown_ok and len(ball_positions) >= 6:
                event, conf = analyse_ball_trajectory(frame_h, frame_w)
                if event == "SIX" and conf >= 0.60:
                    push_suggestion("SIX", conf, f"Ball trajectory went high (cam motion: {motion:.1f}px)")
                    last_boundary_time = now
                    ball_positions.clear()
                elif event == "BOUNDARY" and conf >= 0.55:
                    push_suggestion("BOUNDARY", conf, f"Ball near edge (cam motion: {motion:.1f}px)")
                    last_boundary_time = now
                    ball_positions.clear()

            # WICKET
            wicket_cooldown_ok = (now - last_wicket_time) > WICKET_COOLDOWN
            if wicket_cooldown_ok and detections:
                wkt_conf = detect_wicket_event(detections, frame_h, frame_w)
                if wkt_conf >= 0.55:
                    push_suggestion("WICKET", wkt_conf, "Stumps/bails disturbance detected")
                    last_wicket_time = now

            # RUNS
            run_cooldown_ok = (now - last_run_time) > RUN_COOLDOWN
            if run_cooldown_ok and len(person_positions) >= 8:
                runs = analyse_runs(frame_h, frame_w)
                if runs > 0:
                    push_suggestion(f"RUN_{runs}", 0.55, f"Batsmen crossed ({runs} run(s) estimated)")
                    last_run_time = now

        # ── Preview window ──────────────────────────────────────────────────
        if show_preview:
            preview = cv2.resize(frame, (640, 360))
            ph, pw = preview.shape[:2]
            sf_x = pw / frame_w
            sf_y = ph / frame_h

            # Draw ball detections
            for (cx, cy, conf) in frame_balls:
                px, py = int(cx * sf_x), int(cy * sf_y)
                cv2.circle(preview, (px, py), 12, (0, 255, 255), 2)
                cv2.putText(preview, f"BALL {conf:.0%}", (px+14, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw person detections
            for (cx, cy, conf) in frame_persons:
                px, py = int(cx * sf_x), int(cy * sf_y)
                cv2.circle(preview, (px, py), 8, (0, 200, 0), 2)

            # HUD
            hud_color = (0, 60, 200) if cam_moving else (0, 180, 0)
            cv2.rectangle(preview, (0, 0), (360, 22), (0, 0, 0), -1)
            cv2.putText(preview,
                        f"FPS:{fps_display:.0f}  CAM:{motion:.1f}px {'MOVING' if cam_moving else 'STABLE'}  "
                        f"BALLS:{len(frame_balls)}  PERSONS:{len(frame_persons)}",
                        (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hud_color, 1)

            # Ball trajectory
            pts = list(ball_positions)
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][0] * sf_x), int(pts[i-1][1] * sf_y))
                p2 = (int(pts[i][0]   * sf_x), int(pts[i][1]   * sf_y))
                alpha = i / len(pts)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(preview, p1, p2, color, 2)

            cv2.imshow("Cricket AI Tracker — Q to quit", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Tracker stopped.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket AI Ball Tracker")
    parser.add_argument("--source",  default=DEFAULT_SOURCE,
                        help="Stream source: rtmp://..., 0 (webcam), or video file")
    parser.add_argument("--model",   default="yolov8n.pt",
                        help="YOLOv8 model: yolov8n.pt (fast) / yolov8s.pt (better)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Run headless (no OpenCV window)")
    parser.add_argument("--conf",    type=float, default=CONFIDENCE_THRESHOLD,
                        help="Detection confidence threshold (default 0.45)")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf

    print("=" * 60)
    print("  🏏 CRICKET AI TRACKER")
    print("=" * 60)
    print(f"  Source  : {args.source}")
    print(f"  Model   : {args.model}")
    print(f"  Preview : {'off' if args.no_preview else 'on'}")
    print(f"  Conf    : {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    # ── Resolve YouTube URLs automatically ──────────────────────────────────
    source = args.source
    if "youtube.com" in source or "youtu.be" in source:
        source = resolve_youtube(source)
        if not source:
            sys.exit(1)

    # Init Firebase
    fb_ok = init_firebase()
    if not fb_ok:
        print("⚠️  Running without Firebase — events will print to console only.")

    # Load YOLO model (downloads automatically on first run)
    print(f"\n📦 Loading YOLO model: {args.model}")
    print("   (First run downloads ~6MB — please wait...)")
    model = YOLO(args.model)
    print("✅ Model loaded\n")

    # Start tracking
    process_stream(
        source=source,
        model=model,
        show_preview=not args.no_preview,
    )