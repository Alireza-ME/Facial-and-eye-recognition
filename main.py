import cv2
import sys
import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(0)

    def timerEvent(self):
        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, cascade_filepath,cascade_eye_filepath,cascade_smile_filepath, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(cascade_filepath)
        self.eye = cv2.CascadeClassifier(cascade_eye_filepath)
        self.smile =  cv2.CascadeClassifier(cascade_smile_filepath)
        self.image = QImage()
        self._border = (0, 255, 0)
        self._width = 2

    def detect_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        faces = self.classifier.detectMultiScale(gray_image, 1.3, 5)
        return faces

    def image_data_slot(self, image_data):
        if (self.width() > self.height()) != (image_data.shape[1] > image_data.shape[0]):
            # Need to rotate image data, the screen / camera is rotated
            image_data = cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces = self.detect_faces(image_data)
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_data, (x, y), (x + w, y + h), self._border, self._width)
            roi = gray[y:y+h, x:x+w]
            roi_img = image_data[y:y+h,x:x+w]
            eyes = self.eye.detectMultiScale(roi,3,5)
          #  sm = self.smile.detectMultiScale(roi,1.7,20)	

            for (ex,ey,ew,eh) in eyes:
            	cv2.rectangle(roi_img, (ex,ey),(ex+ew,ey+eh),(255,0,0),3)
#            	   
#        for (mx, my, mw, mh) in sm:
#        	cv2.rectangle(roi_img,(mx, my),(mx+mw, my+mh), (255, 0, 130), 2)

        self.image = self.get_qimage(image_data)
        self.update()

    def get_qimage(self, image):
        height, width, colors = image.shape
        image = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        w = self.width()
        h = self.height()
        cw = self.image.width()
        ch = self.image.height()

        # Keep aspect ratio
        if ch != 0 and cw != 0:
            w = min(cw * h / ch, w)
            h = min(ch * w / cw, h)
            w, h = int(w), int(h)

        painter.drawImage(QtCore.QRect(0, 0, w, h), self.image)
        self.image = QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath,haarcascade_eye_filepath,haarcascade_smile_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        fe = haarcascade_eye_filepath
        fm = haarcascade_smile_filepath
        self.face_detection_widget = FaceDetectionWidget(fp,fe,fm)
        # 1 is used for frontal camera
        self.record_video = RecordVideo(0)
        self.record_video.image_data.connect(self.face_detection_widget.image_data_slot)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.face_detection_widget)
        self.setLayout(layout)


app = QtWidgets.QApplication(sys.argv)
haar_cascade_filepath = cv2.data.haarcascades + '/haarcascade_frontalface_default.xml'
haar_cascade_eye_filepath = cv2.data.haarcascades + '/haarcascade_eye.xml'
haar_cascade_smile_filepath = cv2.data.haarcascades + '/haarcascade_smile.xml'
main_window = QtWidgets.QMainWindow()
main_widget = MainWidget(haar_cascade_filepath,haar_cascade_eye_filepath,haar_cascade_smile_filepath)
main_window.setCentralWidget(main_widget)
main_window.show()
sys.exit(app.exec_())
