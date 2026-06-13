import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

image_path = "images.jpg"  # replace with your image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
    else:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            )

        cv2.imshow("Face Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()