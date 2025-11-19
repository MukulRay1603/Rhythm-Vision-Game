import time
import csv
import random
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


class CentroidTracker:
    def __init__(self, max_lost=8):
        self.next_id = 0
        self.boxes = {}
        self.lost = {}
        self.trails = {}
        self.max_lost = max_lost

    def _centroid(self, box):
        x, y, w, h = box
        return np.array([x + w / 2.0, y + h / 2.0])

    def update(self, detections):
        if len(self.boxes) == 0:
            for box in detections:
                self._register(box)
            return self.boxes, self.trails

        ids = list(self.boxes.keys())
        old_centroids = [self._centroid(self.boxes[i]) for i in ids]
        new_centroids = [self._centroid(b) for b in detections]

        if len(detections) == 0:
            for tid in ids:
                self.lost[tid] += 1
            self._cleanup()
            return self.boxes, self.trails

        dist = np.zeros((len(old_centroids), len(new_centroids)))
        for r, oc in enumerate(old_centroids):
            for c, nc in enumerate(new_centroids):
                dist[r, c] = np.linalg.norm(oc - nc)

        used_rows, used_cols = set(), set()

        while True:
            r, c = np.unravel_index(np.argmin(dist), dist.shape)
            if dist[r, c] == np.inf or dist[r, c] > 80:
                break
            tid = ids[r]
            self.boxes[tid] = detections[c]
            self.lost[tid] = 0
            self.trails[tid].append(new_centroids[c])
            used_rows.add(r)
            used_cols.add(c)
            dist[r, :] = np.inf
            dist[:, c] = np.inf

        for c, box in enumerate(detections):
            if c not in used_cols:
                self._register(box)

        for r, tid in enumerate(ids):
            if r not in used_rows:
                self.lost[tid] += 1

        self._cleanup()
        return self.boxes, self.trails

    def _register(self, box):
        tid = self.next_id
        self.boxes[tid] = box
        self.lost[tid] = 0
        self.trails[tid] = deque(maxlen=60)
        self.trails[tid].append(self._centroid(box))
        self.next_id += 1

    def _cleanup(self):
        dead = [tid for tid in self.boxes if self.lost[tid] > self.max_lost]
        for tid in dead:
            del self.boxes[tid]
            del self.lost[tid]
            del self.trails[tid]


def draw_xp_bar(frame, xp, max_xp, level):
    h = 35
    W = frame.shape[1]

    cv2.rectangle(frame, (0, 0), (W, h), (40, 40, 40), -1)

    fill = int((xp / max_xp) * W)
    color = (255, 120, 255)
    cv2.rectangle(frame, (0, 0), (fill, h), color, -1)

    text = f"LEVEL {level}   XP: {xp}/{max_xp}"
    cv2.putText(frame, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    mp_face = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error.")
        return

    W = int(cap.get(3))
    H = int(cap.get(4))

    tracker = CentroidTracker()

    prev_centers = {}
    prev_areas = {}
    beat_effects = []

    # XP System
    xp = 0
    max_xp = 200
    level = 1
    last_xp_gain = []

    # Combo mechanic
    combos = {}

    # Game prompts
    beat_types = ["MOVE LEFT", "MOVE RIGHT", "LEAN IN", "LEAN BACK"]
    current_prompt = random.choice(beat_types)
    prompt_timer = time.time()

    logfp = open("face_logs.csv", "w", newline="")
    writer = csv.writer(logfp)
    writer.writerow(["timestamp_ms", "face_id", "x", "y", "w", "h"])

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = fd.process(rgb)

            detections = []
            if out.detections:
                for det in out.detections:
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * W)
                    y = int(bb.ymin * H)
                    w = int(bb.width * W)
                    h = int(bb.height * H)
                    detections.append((x, y, w, h))

            boxes, trails = tracker.update(detections)

            ts = int(time.time() * 1000)

            # Draw XP Bar
            draw_xp_bar(frame, xp, max_xp, level)

            # Show prompt (middle of screen)
            cv2.putText(frame, current_prompt,
                        (W//2 - 150, H//2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 3)

            # Change prompt every 4 seconds
            if time.time() - prompt_timer > 4:
                current_prompt = random.choice(beat_types)
                prompt_timer = time.time()

            # process each detected face
            for fid, (x, y, w, h) in boxes.items():
                cx = x + w/2
                cy = y + h/2
                area = w * h

                prev_c = prev_centers.get(fid, np.array([cx, cy]))
                prev_a = prev_areas.get(fid, area)

                dx = cx - prev_c[0]
                dy = cy - prev_c[1]
                dA = area - prev_a

                prev_centers[fid] = np.array([cx, cy])
                prev_areas[fid] = area

                # draw box + id
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 200), 2)
                cv2.putText(frame, f"ID {fid}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

                # Draw motion trails
                pts = list(trails[fid])
                for i in range(1, len(pts)):
                    cv2.line(frame, tuple(pts[i-1].astype(int)),
                             tuple(pts[i].astype(int)),
                             (255, 200, 200), 2)

                # Check if user performed correct action
                performed = None
                if dx < -6:
                    performed = "MOVE LEFT"
                elif dx > 6:
                    performed = "MOVE RIGHT"
                elif dA > 2500:
                    performed = "LEAN IN"
                elif dA < -2500:
                    performed = "LEAN BACK"

                if performed == current_prompt:
                    xp += 20
                    combos[fid] = combos.get(fid, 0) + 1

                    last_xp_gain.append([int(cx), int(cy), time.time()])

                    beat_effects.append({
                        "center": (int(cx), int(cy)),
                        "radius": 10,
                        "max": 200,
                        "color": (255, 255, 255)
                    })

                    current_prompt = random.choice(beat_types)
                    prompt_timer = time.time()

                    if xp >= max_xp:
                        xp = 0
                        level += 1

            # XP floating text
            new_float = []
            for fx, fy, t0 in last_xp_gain:
                age = time.time() - t0
                if age < 1:
                    y_offset = int(fy - age * 40)
                    cv2.putText(frame, "+20 XP", (fx, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (50, 255, 50), 2)
                    new_float.append([fx, fy, t0])
            last_xp_gain = new_float

            # Beat ripple animation
            updated = []
            for eff in beat_effects:
                cx, cy = eff["center"]
                r = eff["radius"]
                if r < eff["max"]:
                    cv2.circle(frame, (cx, cy), int(r),
                               eff["color"], 2)
                    eff["radius"] += 5
                    updated.append(eff)
            beat_effects = updated

            cv2.imshow("Rhythm Game Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    logfp.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
