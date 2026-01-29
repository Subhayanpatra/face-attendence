import cv2
import mediapipe as mp
import numpy as np

from utils.helpers import load_embeddings, match_face
from attendance import mark_attendance
from anti_spoof import is_live_face

# ---------------- MediaPipe Setup ----------------
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

# ---------------- Load Face Database ----------------
database = load_embeddings()

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
frame_id = 0

# 🔒 Session control
attendance_marked = False   # prevents multiple CSV writes
last_action = None          # stores "Punch In" / "Punch Out"

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_id)

    # Default UI
    label = "Waiting"
    color = (0, 255, 255)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        emb = np.array([[p.x, p.y, p.z] for p in landmarks]).flatten()
        name = match_face(emb, database)

        if name and is_live_face(landmarks):

            # ✅ Mark attendance ONLY ONCE per run
            if not attendance_marked:
                last_action = mark_attendance(name)  # Punch In / Punch Out
                attendance_marked = True

            # ✅ Always display real action
            label = f"{name} - {last_action}"

            if last_action == "Punch In":
                color = (0, 255, 0)
            elif last_action == "Punch Out":
                color = (255, 165, 0)
            else:
                color = (200, 200, 200)

        elif name:
            label = f"{name},Blink Eye"
            color = (0, 0, 255)

        else:
            label = "Unknown"
            color = (255, 0, 0)

    # ---------------- Display ----------------
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()