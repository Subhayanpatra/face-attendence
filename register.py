import cv2
import mediapipe as mp
import numpy as np
from utils.helpers import save_embeddings

# ---- MediaPipe Tasks Setup ----
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ---- Registration ----
name = input("Enter user name: ").strip()

cap = cv2.VideoCapture(0)

embeddings = []
frame_id = 0
capturing = False   # 🔑 CONTROL FLAG

print("Press 'S' to START capturing")
print("Press 'Q' to STOP and save")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_id)

    # ---------------- UI MESSAGE ----------------
    if not capturing:
        status_text = "Press S to start capture"
        color = (0, 255, 255)
    else:
        status_text = f"Capturing... Samples: {len(embeddings)}"
        color = (0, 255, 0)

    # ---------------- Capture Logic ----------------
    if capturing and result.face_landmarks:
        landmarks = result.face_landmarks[0]

        embedding = np.array(
            [[p.x, p.y, p.z] for p in landmarks]
        ).flatten()

        embeddings.append(embedding)

    # ---------------- Display ----------------
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        capturing = True

    elif key == ord("q"):
        break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()

# ---------------- Save ----------------
if embeddings:
    save_embeddings(name, embeddings)
    print(f"[SUCCESS] {name} registered with {len(embeddings)} samples.")
else:
    print("[WARNING] No samples captured. Registration failed.")