from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
)
from .image_display import ImageDisplay
from .matplotlib_canvas import MatplotlibCanvas
from .helper_functions import process_data
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contour-finding and Close-packing GUI")
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Panel A
        panelA = QWidget()
        panelA.setObjectName("panelA")
        panelA_layout = QVBoxLayout()
        panelA_layout.setSpacing(5)

        # A-1: Image Display
        self.image_display = ImageDisplay()
        panelA_layout.addWidget(self.image_display, stretch=1)

        # A-2: Processed Results Display
        self.canvas = MatplotlibCanvas()
        panelA_layout.addWidget(self.canvas, stretch=1)

        panelA.setLayout(panelA_layout)

        # Panel B
        panelB = QWidget()
        panelB.setObjectName("panelB")
        panelB_layout = QVBoxLayout()
        panelB_layout.setSpacing(10)

        # B-1: Contour-finding parameters
        b1_group = QGroupBox("Contour-finding Parameters")
        b1_layout = QFormLayout()
        self.param1 = QLineEdit()
        self.param2 = QLineEdit()
        b1_layout.addRow("Parameter 1:", self.param1)
        b1_layout.addRow("Parameter 2:", self.param2)
        b1_group.setLayout(b1_layout)

        # B-2: Close packing parameters
        b2_group = QGroupBox("Close Packing Parameters")
        b2_layout = QFormLayout()
        self.param3 = QLineEdit()
        self.param4 = QLineEdit()
        b2_layout.addRow("Parameter 3:", self.param3)
        b2_layout.addRow("Parameter 4:", self.param4)
        b2_group.setLayout(b2_layout)

        # B-3: Buttons
        b3_group = QGroupBox("Controls")
        b3_layout = QVBoxLayout()
        self.process_button = QPushButton("Process Image")
        b3_layout.addWidget(self.process_button)
        b3_group.setLayout(b3_layout)

        panelB_layout.addWidget(b1_group)
        panelB_layout.addWidget(b2_group)
        panelB_layout.addWidget(b3_group)
        panelB.setLayout(panelB_layout)

        # Panel C: Output Log
        panelC = QWidget()
        panelC.setObjectName("panelC")
        panelC_layout = QVBoxLayout()
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        panelC_layout.addWidget(self.output_log)
        panelC.setLayout(panelC_layout)

        # Add panels to main layout
        main_layout.addWidget(panelA, stretch=1)
        main_layout.addWidget(panelB, stretch=1)
        main_layout.addWidget(panelC, stretch=1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Connect signals
        self.process_button.clicked.connect(self.process_image)

    def process_image(self):
        if self.image_display.image:
            self.output_log.append("Processing image...")
            # Load the image path
            image_path = self.image_display.image
            # Retrieve parameters
            param1 = self.param1.text()
            param2 = self.param2.text()
            param3 = self.param3.text()
            param4 = self.param4.text()
            # Call your processing functions
            pass
