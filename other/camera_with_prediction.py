import flet as ft
import base64
import cv2
from PIL import Image
from consts import THICKNESS, FONT, FONT_SCALE, CLASS_NAMES, EMOTION_COLORS


class CameraWithPrediction(ft.UserControl):
    def __init__(self, model, face_detector, path=None):
        super().__init__()
        self.model = model
        self.face_detector = face_detector
        self.mounted = False
        if path is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(path)

    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        while self.mounted:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = self.face_detector(gray_frame)
                # Iterate over detected faces
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    # Crop face region from frame
                    face_image = frame[y:y+h, x:x+w]
                    if face_image is None or face_image.size == 0:
                        continue
                    pil_image = Image.fromarray(
                        cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                    results = self.model(pil_image)
                    if len(results) == 0 or len(results[0].boxes) == 0:
                        continue
                    max_box = max(results[0].boxes,
                                  key=lambda box: box.conf.tolist()[0])
                    cls = int(max_box.cls[0])
                    confidence = max_box.conf.tolist()[0]
                    confidenceInPercentage = confidence * 100
                    org = [x, y - 5]

                    # Perform emotion prediction
                    cv2.rectangle(frame, (x, y),
                                  (x + w, y + h), EMOTION_COLORS[cls], THICKNESS)
                    cv2.putText(frame, f"{CLASS_NAMES[cls]} {confidenceInPercentage:.0f}%", org,
                                FONT, FONT_SCALE, EMOTION_COLORS[cls], THICKNESS)

            _, im_arr = cv2.imencode('.png', frame)
            im_b64 = base64.b64encode(im_arr)
            self.img.src_base64 = im_b64.decode("utf-8")
            if self.mounted:
                self.update()

    def build(self):
        self.mounted = True
        _, frame = self.cap.read()
        _, im_arr = cv2.imencode('.png', frame)
        im_b64 = base64.b64encode(im_arr)
        self.loading = ft.ProgressRing()
        self.row = ft.Row([self.loading])

        self.img = ft.Image(
            border_radius=ft.border_radius.all(20),
            src_base64=im_b64.decode("utf-8"),
            height=550,
        )

        return self.img

    def _dispose(self):
        self.mounted = False
        self.cap.release()
        return super()._dispose()
