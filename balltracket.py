import cv2
import numpy as np
import streamlink

# 1. Setup Stream
video_url = 'https://www.youtube.com/watch?v=VObJukizpxY'
try:
    streams = streamlink.streams(video_url)
    url = streams["480p"].url if "480p" in streams else streams["best"].url
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2048 * 1000)
except:
    print("Stream Error"); exit()

# List to store previous positions (creates the 'trail')
pts = []

print("TRACKING: Drawing red circle on tiny moving ball...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # STEP 1: ISOLATE RED COLOR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Range for Red (Adjusted to be slightly wider for motion blur)
    lower_red = np.array([0, 130, 100])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # STEP 2: MOTION MASK (To ignore stationary red objects)
    # We only care about the red that is MOVING
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # STEP 3: FIND CONTOURS
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        # Find the smallest object that fits our 'ball' criteria
        for c in cnts:
            area = cv2.contourArea(c)
            if 2 < area < 100: # Tiny object size
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # DRAW THE RED CIRCLE
                # (frame, center, radius, color, thickness)
                cv2.circle(frame, center, int(radius) + 5, (0, 0, 255), 2)
                cv2.putText(frame, "BALL", (int(x), int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # STEP 4: DRAW THE TRAIL (Optional but very helpful for small balls)
    pts.append(center)
    if len(pts) > 20: pts.pop(0) # Keep only last 20 positions
    
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None: continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)

    # Display Result
    cv2.imshow("Red Ball Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()