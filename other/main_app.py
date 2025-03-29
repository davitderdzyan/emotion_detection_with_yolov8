import math
import flet as ft
import cv2
import base64
from ultralytics import YOLO
from consts import THICKNESS, FONT, FONT_SCALE, CLASS_NAMES, EMOTION_COLORS, WEIGHTS_PATH
import dlib
from PIL import Image
from camera_with_prediction import CameraWithPrediction


class MainApp(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.model = YOLO(WEIGHTS_PATH)
        self.face_detector = dlib.get_frontal_face_detector()
        self.camera_container_row = ft.Row(
            [], alignment=ft.MainAxisAlignment.SPACE_AROUND, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.file_picker = ft.FilePicker(
            on_result=lambda val: self.file_picker_result(val), on_upload=lambda val: self.on_upload_progress(val))
        self.loading = False
        self.camera_open = False

    def file_picker_result(self, e: ft.FilePickerResultEvent):
        try:
            if e.files is not None:
                if e.files[0].path.endswith('.mp4') or e.files[0].path.endswith('.mov'):
                    self.camera_container_row.controls = [
                        ft.Container(CameraWithPrediction(model=self.model, face_detector=self.face_detector, path=e.files[0].path))]
                    self.loading = False
                    self.camera_open = False
                    self.camera_container_row.update()
                    return
                # self.model(e.files[0].path)
                image = cv2.imread(e.files[0].path)
                gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = self.face_detector(gray_frame)
                # Iterate over detected faces
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_image = image[y:y+h, x:x+w]
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
                    cv2.rectangle(image, (x, y),
                                  (x + w, y + h), EMOTION_COLORS[cls], THICKNESS)
                    cv2.putText(image, f"{CLASS_NAMES[cls]} {confidenceInPercentage:.0f}%", org,
                                FONT, FONT_SCALE, EMOTION_COLORS[cls], THICKNESS)
                if image is None:
                    self.camera_container_row.controls = []
                    self.loading = False
                    self.camera_container_row.update()
                    return
                _, im_arr = cv2.imencode('.png', image)
                im_b64 = base64.b64encode(im_arr)
                self.loading = False
                self.camera_container_row.controls = [ft.Container(
                    ft.Image(
                        src_base64=im_b64.decode("utf-8"),
                        height=450,
                        border_radius=ft.border_radius.all(20),
                    ),
                )]
                self.camera_container_row.update()
            else:
                self.camera_container_row.controls = []
                self.loading = False
                self.camera_container_row.update()
                return
        except Exception as e:
            print('Something went wrong.')
            print(e)
        finally:
            self.camera_container_row.controls = []
            self.loading = False

    def on_upload_progress(e: ft.FilePickerUploadEvent):
        return

    def on_pick_image(self, val):
        if self.loading:
            return
        self.file_picker.pick_files(allow_multiple=False, allowed_extensions=[
            'jpg', 'jpeg', 'png'])
        self.loading = True
        self.camera_open = False
        self.camera_container_row.controls = [
            ft.ProgressRing(color='black')
        ]
        self.camera_container_row.update()

    def on_pick_video(self, val):
        if self.loading:
            return
        self.file_picker.pick_files(allow_multiple=False, allowed_extensions=[
            'mp4', 'mov'])
        self.loading = True
        self.camera_open = False
        self.camera_container_row.controls = [
            ft.ProgressRing(color='black')
        ]
        self.camera_container_row.update()

    def on_camera_open(self, val):
        if self.loading or self.camera_open:
            return
        self.loading = True
        self.camera_container_row.controls = [
            ft.ProgressRing(color='black')
        ]
        self.camera_container_row.update()
        self.camera_container_row.controls = [
            ft.Container(CameraWithPrediction(model=self.model, face_detector=self.face_detector))]
        self.loading = False
        self.camera_open = True
        self.camera_container_row.update()

    def build(self):
        return ft.Container(
            margin=ft.margin.only(bottom=40),
            content=ft.ResponsiveRow([
                ft.Card(
                    elevation=15,
                    content=ft.ResponsiveRow([
                        ft.Column(col=12, controls=[
                            ft.Container(
                                width=math.inf,
                                bgcolor=ft.colors.BLUE_700,
                                padding=10,
                                border_radius=ft.border_radius.all(20),
                                content=ft.Column(
                                    alignment=ft.MainAxisAlignment.CENTER,
                                    controls=[
                                        ft.ResponsiveRow(
                                            [
                                                ft.Container(
                                                    content=ft.Text(
                                                        "Emotion Detection",
                                                        size=20, weight="bold",
                                                        text_align=ft.TextAlign.CENTER,
                                                        color=ft.colors.WHITE
                                                    )
                                                ),
                                            ]
                                        ),

                                        ft.Row(

                                            [
                                                ft.ElevatedButton(
                                                    text="Open Camera",
                                                    width=300,
                                                    on_click=lambda val: self.on_camera_open(val)),

                                                ft.ElevatedButton(
                                                    text="Pick Image",
                                                    width=300,
                                                    on_click=lambda val: self.on_pick_image(val)),
                                                ft.ElevatedButton(
                                                    text="Pick Video",
                                                    width=300,
                                                    on_click=lambda val: self.on_pick_video(val)),

                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        self.camera_container_row,
                                        self.file_picker

                                    ]
                                ),
                            )
                        ])

                    ])


                ),
            ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )

    def _dispose(self):
        self.cap.release()
        return super()._dispose()
