import numpy as np

def normalize_xy(xy21):
    """Center and scale-normalize a 21×2 landmark array."""
    xy = xy21.copy()
    wrist = xy[0]
    xy -= wrist
    scale = np.linalg.norm(xy).mean() + 1e-6
    return xy / scale

def seq_from_landmarks(landmarks_list, W, H):
    """Convert MediaPipe landmark objects → (T, 21, 2) numpy array."""
    seq = []
    for lm in landmarks_list:
        xy = np.array([[p.x*W, p.y*H] for p in lm.landmark], dtype=np.float32)
        seq.append(normalize_xy(xy))
    return np.stack(seq, axis=0)
