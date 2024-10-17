from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2


class ImageDisplay(QWidget):
    image_loaded_signal = pyqtSignal()
    point_clicked_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        # set the size
        # self.setFixedSize(400, 300)

        self.setObjectName("image_display")
        self.label = QLabel("Drop an image here")
        self.label.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # install event filter on the label to capture mouse events
        # self.label.installEventFilter(self)
        self.image_path = None  # image path
        self.image = None  # Store the image data
        self.pixmap = None  # Store the pixmap

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
            self.image_path = filepath  # Store the image path

    def display_image(self, filepath):
        image_bgr = cv2.imread(filepath)
        if image_bgr is not None:
            self.image = image_bgr  # store the image data
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytesPerLine = 3 * width
            qImg = QImage(
                image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qImg)
            # scale the pixmap
            pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)
            self.pixmap = pixmap

            # Emit signal to notify that the image has been loaded
            self.image_loaded_signal.emit()
        else:
            self.label.setText("Failed to load image")

    def eventFilter(self, source, event):
        if source == self.label and event.type() == Qt.MouseButtonPress:
            self.mousePressEvent(event)
            return True
        return super().eventFilter(source, event)

    def mousePressEvent(self, event):
        if self.image is not None and self.label.pixmap():
            # Map the click position to image coordinates
            x = event.pos().x()
            y = event.pos().y()

            label_width = self.width()
            label_height = self.height()
            pixmap = self.pixmap
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()

            # Compute scaling factors and offsets
            if pixmap_width * label_height > pixmap_height * label_width:
                # Image scaled to fit label width
                scale = pixmap_width / label_width
                scaled_width = label_width
                scaled_height = pixmap_height / scale
                offset_x = 0
                offset_y = (label_height - scaled_height) / 2
            else:
                # Image scaled to fit label height
                scale = pixmap_height / label_height
                scaled_width = pixmap_width / scale
                scaled_height = label_height
                offset_x = (label_width - scaled_width) / 2
                offset_y = 0

            # Adjust coordinates to image coordinate system
            if (offset_x <= x <= offset_x + scaled_width) and (
                offset_y <= y <= offset_y + scaled_height
            ):
                image_x = int((x - offset_x) * (self.image.shape[1] / scaled_width))
                image_y = int((y - offset_y) * (self.image.shape[0] / scaled_height))
                # Emit signal with image coordinates
                self.point_clicked_signal.emit(image_x, image_y)

    def get_pixel_value(self, x, y):
        # Get BGR value from image at position (x, y)
        if self.image is not None:
            h, w, _ = self.image.shape
            print(h, w)
            if 0 <= y < h and 0 <= x < w:
                bgr = self.image[y, x]
                return bgr.tolist()  # Convert to list for easier handling
        return None
