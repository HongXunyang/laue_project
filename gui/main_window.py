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

        # Panel A
        panelA = QWidget()
        panelA_layout = QVBoxLayout()

        # A-1: Image Display
        self.image_display = ImageDisplay()
        panelA_layout.addWidget(self.image_display)

        # A-2: Processed Results Display
        self.canvas = MatplotlibCanvas()
        panelA_layout.addWidget(self.canvas)

        panelA.setLayout(panelA_layout)

        # Panel B
        panelB = QWidget()
        panelB_layout = QVBoxLayout()

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
        panelC_layout = QVBoxLayout()
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        panelC_layout.addWidget(self.output_log)
        panelC.setLayout(panelC_layout)

        # Add panels to main layout
        main_layout.addWidget(panelA)
        main_layout.addWidget(panelB)
        main_layout.addWidget(panelC)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Connect signals
        self.process_button.clicked.connect(self.process_image)

    def process_image(self):
        # Placeholder for processing
        self.output_log.append("Processing image...")
        # Integrate your processing functions here

        data = process_data()
        self.canvas.plot(data)
        self.output_log.append("Processing complete.")
