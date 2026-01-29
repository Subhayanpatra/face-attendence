import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / (2.0 * horizontal)

def is_live_face(landmarks):
    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    ear = (ear_left + ear_right) / 2
    return ear < 0.20   # blink detected