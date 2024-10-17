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

        # variables
        self.selection_state = None
        self.selected_points = []

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
        self.matplotlib_canvas = MatplotlibCanvas()
        panelA_layout.addWidget(self.matplotlib_canvas, stretch=1)

        panelA.setLayout(panelA_layout)

        # Panel B
        panelB = QWidget()
        panelB.setObjectName("panelB")
        panelB_layout = QVBoxLayout()
        panelB_layout.setSpacing(10)

        # B-1: Contour-finding parameters
        contour_finding_params = QGroupBox("contour_finding_params")
        contour_finding_params_layout = QFormLayout()
        self.param1 = QLineEdit()
        self.param2 = QLineEdit()
        contour_finding_params_layout.addRow("Parameter 1:", self.param1)
        contour_finding_params_layout.addRow("Parameter 2:", self.param2)
        contour_finding_params.setLayout(contour_finding_params_layout)

        # B-2: Close packing parameters
        close_packing_params = QGroupBox("close_packing_params")
        close_packing_params_layout = QFormLayout()
        self.param3 = QLineEdit()
        self.param4 = QLineEdit()
        close_packing_params_layout.addRow("Parameter 3:", self.param3)
        close_packing_params_layout.addRow("Parameter 4:", self.param4)
        close_packing_params.setLayout(close_packing_params_layout)

        # B-3: Buttons
        controls = QGroupBox("controls")
        controls_layout = QVBoxLayout()

        # Process Button
        self.process_button = QPushButton("Process Image")
        controls_layout.addWidget(self.process_button)

        # select-points button
        self.select_points_button = QPushButton("Select Points")
        controls_layout.addWidget(self.select_points_button)

        controls.setLayout(controls_layout)

        panelB_layout.addWidget(contour_finding_params)
        panelB_layout.addWidget(close_packing_params)
        panelB_layout.addWidget(controls)
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
        self.select_points_button.clicked.connect(self.start_point_selection)
        # -----------------------
        # Signal management
        # -----------------------
        self.image_display.image_loaded_signal.connect(self._on_image_loaded)
        self.image_display.point_clicked_signal.connect(self._on_image_clicked)

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

    def start_point_selection(self):
        if self.image_display.image is not None:
            self.selection_state = "selecting_stripe_points"
            self.selected_points = []
            self.output_log.append("Please select three stripe points on the image.")
        else:
            self.output_log.append("Please load an image first.")

    # -----------------------
    # Signal management methods
    # -----------------------

    def _on_image_loaded(self):
        self.output_log.append("Image loaded successfully")

    def _on_image_clicked(self, x, y):
        if self.selection_state == "selecting_stripe_points":
            bgr = self.image_display.get_pixel_value(x, y)
            if bgr is not None:
                self.selected_points.append(
                    {"type": "stripe", "position": (x, y), "bgr": bgr}
                )
                self.output_log.append(
                    f"Selected stripe point at ({x}, {y}), BGR: {bgr}"
                )
                if len([p for p in self.selected_points if p["type"] == "stripe"]) == 3:
                    self.selection_state = "selecting_background_points"
                    self.output_log.append(
                        "Please select three background points on the image."
                    )
            else:
                self.output_log.append("Invalid point selected.")
        elif self.selection_state == "selecting_background_points":
            bgr = self.image_display.get_pixel_value(x, y)
            if bgr is not None:
                self.selected_points.append(
                    {"type": "background", "position": (x, y), "bgr": bgr}
                )
                self.output_log.append(
                    f"Selected background point at ({x}, {y}), BGR: {bgr}"
                )
                if (
                    len([p for p in self.selected_points if p["type"] == "background"])
                    == 3
                ):
                    self.selection_state = None
                    self.output_log.append("Point selection completed.")
                    self.handle_selected_points()
            else:
                self.output_log.append("Invalid point selected.")

    def handle_selected_points(self):
        # Process the selected points as needed
        stripe_points = [p for p in self.selected_points if p["type"] == "stripe"]
        background_points = [
            p for p in self.selected_points if p["type"] == "background"
        ]
        self.output_log.append("Stripe points:")
        for p in stripe_points:
            self.output_log.append(f"Position: {p['position']}, BGR: {p['bgr']}")
        self.output_log.append("Background points:")
        for p in background_points:
            self.output_log.append(f"Position: {p['position']}, BGR: {p['bgr']}")
        # You can now use these points for further processing
