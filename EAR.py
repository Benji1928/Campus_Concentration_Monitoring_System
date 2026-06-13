import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(points):
    # points order: left corner, upper1, upper2, right corner, lower1, lower2
    horizontal = distance(points[0], points[3])
    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def get_point(landmarks, index, width, height):
    lm = landmarks[index]
    return int(lm.x * width), int(lm.y * height)

image_path = "images.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

height, width = image.shape[:2]
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
    else:
        face = results.multi_face_landmarks[0].landmark

        left_eye_points = [get_point(face, idx, width, height) for idx in LEFT_EYE]
        right_eye_points = [get_point(face, idx, width, height) for idx in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        print(f"Left EAR: {left_ear:.3f}")
        print(f"Right EAR: {right_ear:.3f}")
        print(f"Average EAR: {avg_ear:.3f}")