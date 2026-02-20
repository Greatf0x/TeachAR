import os, time, math, collections
import numpy as np
import cv2
import joblib
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

from features import seq_from_landmarks

# ---------------- Config ----------------
T = 32
SMOOTH_K = 5
SHOW_CONF = True
CAM_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

MODEL_PATH = "data/models/knn.pkl"

# ---------------- Finger config ----------------
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
current_finger = "index"

# ---------------- Drawing config ----------------
class Smoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.has = False
        self.x = 0.0
        self.y = 0.0
    def update(self, x, y):
        if not self.has:
            self.x, self.y = float(x), float(y)
            self.has = True
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return int(self.x), int(self.y)

coord_smoother = Smoother(alpha=0.4)
min_move = 2

# White first
COLORS = [
    (255, 255, 255),  # WHITE
    (0, 0, 255),      # RED
    (255, 0, 0),      # BLUE
    (0, 255, 255),    # YELLOW
]
COLOR_NAMES = {0: "WHITE", 1: "RED", 2: "BLUE", 3: "YELLOW"}
color_index = 0
current_color = COLORS[color_index]

ERASER_RADIUS = 20
eraser_mode = False

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ---------------- Drawing state ----------------
strokes = []
saved_strokes = []

pen_down = False
up_frames = 0
down_frames = 0
UP_THRESH = 4
DOWN_THRESH = 4

def finger_up(landmarks, finger_name, H):
    tip_id = FINGER_TIPS[finger_name]
    pip_id = FINGER_PIPS[finger_name]
    y_tip = landmarks[tip_id].y * H
    y_pip = landmarks[pip_id].y * H
    return y_tip < y_pip

def is_pinch_raw(landmarks, W, H, thresh_px=40):
    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    index_tip = landmarks[FINGER_TIPS["index"]]
    x1, y1 = thumb_tip.x * W, thumb_tip.y * H
    x2, y2 = index_tip.x * W, index_tip.y * H
    return math.hypot(x2 - x1, y2 - y1) < thresh_px

def start_new_stroke():
    strokes.append({"points": [], "color": current_color})

def add_point(x, y):
    if not strokes:
        start_new_stroke()
    pts = strokes[-1]["points"]
    if (not pts) or (abs(pts[-1][0] - x) + abs(pts[-1][1] - y) > min_move):
        pts.append((x, y))

def erase_strokes_at(x, y, radius):
    global strokes
    new_strokes = []
    r2 = radius * radius
    for stroke in strokes:
        pts = stroke["points"]
        kept = []
        for (px, py) in pts:
            dx = px - x
            dy = py - y
            if dx * dx + dy * dy > r2:
                kept.append((px, py))
        if len(kept) >= 2:
            new_strokes.append({"points": kept, "color": stroke["color"]})
    strokes = new_strokes

def render_all_strokes(img):
    for stroke in saved_strokes:
        pts = stroke["points"]
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (120, 120, 120), 2)

    for stroke in strokes:
        pts = stroke["points"]
        if len(pts) < 2:
            continue
        color = stroke["color"]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 2)

def export_png(frame_shape):
    H, W = frame_shape[:2]
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    render_all_strokes(canvas)
    fname = f"drawing_{int(time.time())}.png"
    path = os.path.join(EXPORT_DIR, fname)
    cv2.imwrite(path, canvas)
    print(f"[EXPORT] Saved drawing to {path}")
    return path

def fix_len(seq, T=32):
    if seq.shape[0] >= T:
        return seq[-T:].reshape(-1)
    pad = np.repeat(seq[-1][None, ...], T - seq.shape[0], axis=0)
    return np.concatenate([seq, pad], axis=0).reshape(-1)

def knn_confidence(knn, x):
    try:
        dists, _ = knn.kneighbors([x], n_neighbors=min(3, knn.n_neighbors), return_distance=True)
        d = float(np.mean(dists))
        return 1.0 / (1.0 + d)
    except Exception:
        return None

# Gesture map
GESTURE_ACTIONS = {
    "swipe_right": "next_slide",
    "swipe_left": "prev_slide",
}

def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")

last_trigger_gesture = None
last_trigger_time = 0.0
GESTURE_COOLDOWN_SEC = 0.8

def apply_gesture_action(gesture, now):
    global last_trigger_gesture, last_trigger_time

    g = normalize_label(gesture)
    action = GESTURE_ACTIONS.get(g)
    if action is None:
        return

    if g == last_trigger_gesture and (now - last_trigger_time) < GESTURE_COOLDOWN_SEC:
        return

    # Try to focus slide window reliably
    w, h = pyautogui.size()
    pyautogui.moveTo(w//2, h//2, duration=0.01)
    pyautogui.click()

    if action == "next_slide":
        pyautogui.press("right")
        print("[ACTION] NEXT SLIDE")
    elif action == "prev_slide":
        pyautogui.press("left")
        print("[ACTION] PREV SLIDE")

    last_trigger_gesture = g
    last_trigger_time = now

def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing model: {MODEL_PATH}\nRun: python train_knn.py")

    knn = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit("Camera failed to open (check permissions / continuity camera).")

    win_name = "TeachAR - Live Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)  # <— window size you can change

    buffer = []
    votes = collections.deque(maxlen=SMOOTH_K)

    draw_mode = False  # D toggles
    pinch_state = False
    pinch_on_frames = 0
    pinch_off_frames = 0
    PINCH_START_FRAMES = 3
    PINCH_END_FRAMES = 3

    global pen_down, up_frames, down_frames, color_index, current_color, eraser_mode, current_finger

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = hands.process(rgb)

            hl = None
            x = y = None

            if out.multi_hand_landmarks:
                hl = out.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame,
                    hl,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )
                tip_id = FINGER_TIPS[current_finger]
                x = int(hl.landmark[tip_id].x * W)
                y = int(hl.landmark[tip_id].y * H)

            # ---- DRAW MODE ----
            if draw_mode and hl is not None:
                if eraser_mode and x is not None and y is not None:
                    erase_strokes_at(x, y, ERASER_RADIUS)
                    pen_down = False
                else:
                    raw_pinch = is_pinch_raw(hl.landmark, W, H)
                    if raw_pinch:
                        pinch_on_frames += 1
                        pinch_off_frames = 0
                    else:
                        pinch_off_frames += 1
                        pinch_on_frames = 0

                    if pinch_on_frames >= PINCH_START_FRAMES:
                        pinch_state = True
                    if pinch_off_frames >= PINCH_END_FRAMES:
                        pinch_state = False

                    # draw when finger is UP (index tip above pip)
                    is_up = finger_up(hl.landmark, current_finger, H)
                    if is_up:
                        up_frames += 1
                        down_frames = 0
                    else:
                        down_frames += 1
                        up_frames = 0

                    if not pen_down and up_frames >= UP_THRESH:
                        pen_down = True
                        start_new_stroke()
                        coord_smoother.has = False

                    if pen_down and down_frames >= DOWN_THRESH:
                        pen_down = False
                        coord_smoother.has = False

                    if pen_down and x is not None and y is not None:
                        sx, sy = coord_smoother.update(x, y)
                        add_point(sx, sy)

            # ---- Classification ----
            if out.multi_hand_landmarks:
                buffer.append(out.multi_hand_landmarks[0])
                if len(buffer) > T:
                    buffer.pop(0)

                if len(buffer) >= 10:
                    seq = seq_from_landmarks(buffer, W, H)
                    feat = fix_len(seq, T=T)
                    pred = knn.predict([feat])[0]
                    votes.append(pred)
                    stable = max(set(votes), key=votes.count)

                    conf = knn_confidence(knn, feat) if SHOW_CONF else None
                    txt = f"Gesture: {stable}"
                    if conf is not None:
                        txt += f" (conf~{conf:.2f})"
                    cv2.putText(frame, txt, (20, 80), FONT, 0.9, (0, 255, 0), 2)

                    # Only slide control when NOT drawing
                    if (not draw_mode) and (conf is None or conf > 0.10):
                        apply_gesture_action(stable, time.time())

            render_all_strokes(frame)

            # HUD
            mode_text = "DRAW" if draw_mode else "GESTURE"
            cv2.putText(
                frame,
                f"Mode: {mode_text} | Color: {COLOR_NAMES[color_index]} | D toggle | Z color | E eraser | P export | ESC quit",
                (20, 40),
                FONT,
                0.5,
                (255, 255, 255),
                2,
            )

            cv2.imshow(win_name, frame)
            k = cv2.waitKey(1) & 0xFF

            if k == 27:
                break
            if k in (ord("d"), ord("D")):
                draw_mode = not draw_mode
                pen_down = False
                coord_smoother.has = False
            if k in (ord("z"), ord("Z")):
                color_index = (color_index + 1) % len(COLORS)
                current_color = COLORS[color_index]
                pen_down = False
                coord_smoother.has = False
            if k in (ord("e"), ord("E")):
                eraser_mode = not eraser_mode
                pen_down = False
                coord_smoother.has = False
            if k in (ord("p"), ord("P")):
                export_png(frame.shape)
            if k == ord("2"):
                current_finger = "index"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()