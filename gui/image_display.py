from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2


class ImageDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("imageDisplay")
        self.label = QLabel("Drop an image here")
        self.label.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.image = None  # Store the image path

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self.display_image(filepath)
            self.image = filepath  # Store the image path

    def display_image(self, filepath):
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)
        else:
            self.label.setText("Failed to load image")
