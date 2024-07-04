import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize the MediaPipe face detection and face mesh modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

def load_glasses_images():
    UPLOAD_FOLDER = 'uploads'
    uploaded_images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.png')]
    glasses_images = {img: {'path': os.path.join(UPLOAD_FOLDER, img), 'x_adjust': 0, 'y_adjust': 0} for img in uploaded_images}
    return glasses_images

glasses_images = load_glasses_images()
glasses_index = 0
current_glasses_key = list(glasses_images.keys())[glasses_index]
glasses = cv2.imread(glasses_images[current_glasses_key]['path'], cv2.IMREAD_UNCHANGED)

def overlay_image(frame, overlay_img, x, y, scale=1.0):
    overlay_width = int(overlay_img.shape[1] * scale)
    overlay_height = int(overlay_img.shape[0] * scale)
    overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))

    y1, y2 = y, y + overlay_resized.shape[0]
    x1, x2 = x, x + overlay_resized.shape[1]

    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return frame

    alpha_s = overlay_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * overlay_resized[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])

    return frame

def overlay_glasses(frame, glasses, left_eye, right_eye):
    # Calculate the width and position of the glasses
    glasses_width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2)
    scale = glasses_width / glasses.shape[1]
    x = int(left_eye[0] - glasses_width / 4)
    y = int(left_eye[1] - glasses.shape[0] * scale / 2)
    
    return overlay_image(frame, glasses, x, y, scale)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection, \
    mp_face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)

        if face_results.detections and mesh_results.multi_face_landmarks:
            for detection in face_results.detections:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    ih, iw, _ = frame.shape

                    # Get key landmarks for positioning the glasses
                    left_eye = (int(face_landmarks.landmark[33].x * iw), int(face_landmarks.landmark[33].y * ih))
                    right_eye = (int(face_landmarks.landmark[263].x * iw), int(face_landmarks.landmark[263].y * ih))

                    frame = overlay_glasses(frame, glasses, left_eye, right_eye)

        cv2.imshow('AR Glasses', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()