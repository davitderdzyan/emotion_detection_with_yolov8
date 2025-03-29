import cv2

# Constants
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
WEIGHTS_PATH = "../runs/detect/train10/weights/best.pt"

# Emotion labels and corresponding colors
CLASS_NAMES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

EMOTION_COLORS = [
    (255, 0, 0),       # Anger (Red)
    (0, 255, 255),     # Contempt (Yellow)
    (0, 0, 255),       # Disgust (Blue)
    (255, 255, 0),     # Fear (Cyan)
    (0, 255, 0),       # Happy (Green)
    (128, 128, 128),   # Neutral (Gray)
    (255, 165, 0),     # Sad (Orange)
    (255, 192, 203)    # Surprise (Pink)
]
