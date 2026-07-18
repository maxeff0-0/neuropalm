import numpy as np

def normalize_landmarks(hand_landmarks):
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts/scale
        
    return pts.flatten()
