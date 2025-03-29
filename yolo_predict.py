import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import dlib

# consts


thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
classNames = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

emotion_colors = [
    (255, 0, 0),       # Anger (Red)
    (0, 255, 255),     # Contempt (Yellow)
    (0, 0, 255),       # Disgust (Blue)
    (255, 255, 0),     # Fear (Cyan)
    (0, 255, 0),       # Happy (Green)
    (128, 128, 128),   # Neutral (Gray)
    (255, 165, 0),     # Sad (Orange)
    (255, 192, 203)    # Surprise (Pink)
]

# Load the trained model
weights_path = "./runs/detect/train10/weights/best.pt"
model = YOLO(weights_path)


face_detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 380)

while True:
    ret, frame = cap.read()

    if ret:
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_detector(gray_frame)
        # Iterate over detected faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Crop face region from frame
            face_image = frame[y:y+h, x:x+w]
            if face_image is None or face_image.size == 0:
                continue
            pil_image = Image.fromarray(
                cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            results = model(pil_image)
            if len(results) == 0 or len(results[0].boxes) == 0:
                continue
            cls = int(results[0].boxes[0].cls[0])
            org = [x, y - 5]
            # Convert OpenCV frame to PIL Image
            color = emotion_colors[cls]

            # Perform emotion prediction
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h), color, thickness)
            cv2.putText(frame, classNames[cls], org,
                        font, fontScale, color, thickness)
        cv2.imshow("Face Detection", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
