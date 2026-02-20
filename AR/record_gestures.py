import cv2, os, time, numpy as np
import mediapipe as mp
from features import seq_from_landmarks

# -------- CONFIG --------
DUR_SEC = 1.5          # length of each recording
COUNTDOWN_SEC = 1.5    # time before recording starts
MIN_FRAMES = 8         # need at least this many frames to save
CAM_INDEX = 0          # laptop cam with AVFoundation
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ask once at start
LABEL = input("Enter gesture label (e.g., fist, swipe_right): ").strip()
SAVE_DIR = f"data/raw/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands

def text(frame, msg, y, color=(255,255,255), scale=0.8, thick=2):
    cv2.putText(frame, msg, (20, y), FONT, scale, color, thick)

def badge(frame, msg, color):
    (w, h), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
    pad = 14
    x0, y0 = 20 - pad, 60 - pad
    x1, y1 = 20 + w + pad, 60 + h + pad
    cv2.rectangle(frame, (x0,y0), (x1,y1), color, -1)
    cv2.putText(frame, msg, (20, 60), FONT, 0.9, (255,255,255), 2)

def clips_count(label_dir):
    if not os.path.isdir(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith(".npy")])

def main():
    global LABEL, SAVE_DIR
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit("Camera failed to open. Check Continuity Camera and permissions.")

    print("Controls: R = record, N = change label, ESC = quit")

    state = "IDLE"        # IDLE | COUNTDOWN | RECORDING | SAVED
    state_t0 = 0.0
    last_key = None
    saved_msg_t0 = 0.0
    saved_msg_dur = 1.2
    clips_done = clips_count(SAVE_DIR)

    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        buf = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = hands.process(rgb)
            H, W = frame.shape[:2]

            if out.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, out.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS
                )

            now = time.time()

            # ---- STATE RENDER ----
            if state == "IDLE":
                text(frame, f"Label: {LABEL}   Clips saved: {clips_done}", 40)
                text(frame, "R = record  |  N = change label  |  ESC = quit", 70)

            elif state == "COUNTDOWN":
                elapsed = now - state_t0
                left = max(0.0, COUNTDOWN_SEC - elapsed)
                badge(frame, f"GET READY {int(np.ceil(left))}", (0,140,255))
                if elapsed >= COUNTDOWN_SEC:
                    state = "RECORDING"
                    state_t0 = now
                    buf = []

            elif state == "RECORDING":
                elapsed = now - state_t0
                left = max(0.0, DUR_SEC - elapsed)
                badge(frame, f"REC \u25CF  {left:0.1f}s", (0,0,255))
                if out.multi_hand_landmarks:
                    buf.append(out.multi_hand_landmarks[0])
                if elapsed >= DUR_SEC:
                    if len(buf) >= MIN_FRAMES:
                        arr = seq_from_landmarks(buf, W, H)
                        fname = f"{int(time.time())}.npy"
                        out_path = os.path.join(SAVE_DIR, fname)
                        np.save(out_path, arr)
                        clips_done += 1
                        state = "SAVED"
                        saved_msg_t0 = now
                    else:
                        state = "IDLE"

            elif state == "SAVED":
                text(frame, f"Label: {LABEL}   Clips saved: {clips_done}", 40)
                badge(frame, "SAVED \u2714", (0,160,0))
                if now - saved_msg_t0 >= saved_msg_dur:
                    state = "IDLE"
                    
            # ---- WINDOW & KEYS ----
            cv2.imshow("TeachAR Recorder", frame)
            k = cv2.waitKey(1) & 0xFF

            if k != 255 and k != last_key:  # debounce
                if k == 27:  # ESC
                    break
                elif k in (ord('r'), ord('R')):
                    if state in ("IDLE", "SAVED"):
                        state = "COUNTDOWN"
                        state_t0 = time.time()
                elif k in (ord('n'), ord('N')) and state == "IDLE":
                    new_label = input("Enter new label: ").strip()
                    if new_label:
                        LABEL = new_label
                        SAVE_DIR = f"data/raw/{LABEL}"
                        os.makedirs(SAVE_DIR, exist_ok=True)
                        clips_done = clips_count(SAVE_DIR)
            last_key = k
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()