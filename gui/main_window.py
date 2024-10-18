import numpy as np
import cv2, json
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
from classes import image2contours, visualize_contours


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contour-finding and Close-packing GUI")
        self.initUI()

        # variables
        self.selection_state = None
        self.selected_points = []
        self.stripes_vectors = []
        self.background_vectors = []
        self.target_background_vector = None
        self.default_params_contours_finding = {
            "epsilon": 2.5,
            "lowercut": 100,
            "gaussian_window": (7, 7),
            "area_lowercut": 2000,
        }

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
        panelA_layout.addWidget(self.image_display, stretch=3)

        # A-2: Processed Results Display
        self.matplotlib_canvas = MatplotlibCanvas()
        panelA_layout.addWidget(self.matplotlib_canvas, stretch=2)

        panelA.setLayout(panelA_layout)

        # Panel B
        panelB = QWidget()
        panelB.setObjectName("panelB")
        panelB_layout = QVBoxLayout()
        panelB_layout.setSpacing(10)

        # B-1: Contour-finding parameters
        contour_finding_params = QGroupBox("contour_finding_params")
        contour_finding_params_layout = QFormLayout()
        self.epsilon_input = QLineEdit()
        self.lowercut_input = QLineEdit()
        self.area_lowercut_input = QLineEdit()
        self.gaussian_size_input = QLineEdit()
        # Set placeholders or default values
        self.epsilon_input.setPlaceholderText("Default: 2.5")
        self.lowercut_input.setPlaceholderText("Default: 100")
        self.area_lowercut_input.setPlaceholderText("Default: 2000")
        self.gaussian_size_input.setPlaceholderText("Default: 7")
        contour_finding_params_layout.addRow("Epsilon:", self.epsilon_input)
        contour_finding_params_layout.addRow("Lowercut:", self.lowercut_input)
        contour_finding_params_layout.addRow("Area lowercut:", self.area_lowercut_input)
        contour_finding_params_layout.addRow(
            "Gauss. filter size:", self.gaussian_size_input
        )

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

        # select-points button
        self.select_points_button = QPushButton("Select Points")
        controls_layout.addWidget(self.select_points_button)

        # Process Button
        self.process_button = QPushButton("Process Image")
        controls_layout.addWidget(self.process_button)

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
        main_layout.addWidget(panelA, stretch=3)
        main_layout.addWidget(panelB, stretch=2)
        main_layout.addWidget(panelC, stretch=2)

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
        if self.image_display.image is not None:
            self.output_log.append("----------- Processing image -----------\n")
            # Retrieve parameters, use defaults if input is empty
            epsilon_text = self.epsilon_input.text()
            lowercut_text = self.lowercut_input.text()
            area_lowercut_text = self.area_lowercut_input.text()
            gaussian_size_text = self.gaussian_size_input.text()

            # Load the image path
            image_path = self.image_display.image_path
            image = self.image_display.image
            rows, columns, channels = image.shape
            number_pixels = rows * columns
            estimated_time = int(number_pixels / (1024 * 2048) * 0.25)
            self.output_log.append(f"Estimated time: {estimated_time} seconds\n")
            epsilon = (
                float(epsilon_text)
                if epsilon_text
                else self.default_params_contours_finding["epsilon"]
            )
            lowercut = (
                int(lowercut_text)
                if lowercut_text
                else self.default_params_contours_finding["lowercut"]
            )
            area_lowercut = (
                int(area_lowercut_text)
                if area_lowercut_text
                else self.default_params_contours_finding["area_lowercut"]
            )
            gaussian_window = (
                (int(gaussian_size_text), int(gaussian_size_text))
                if gaussian_size_text
                else self.default_params_contours_finding["gaussian_window"]
            )
            cv2.waitKey(0)
            contours, approximated_contours, hulls = image2contours(
                image,
                stripes_vectors=self.stripes_vectors,
                background_vectors=self.background_vectors,
                epsilon=epsilon,
                lowercut=lowercut,
                area_lowercut=area_lowercut,
                gaussian_window=gaussian_window,
                is_gaussian_filter=True,
            )
            image_to_visualize = visualize_contours(
                image, approximated_contours, hulls, is_plot=False
            )
            # detele later
            min_area = min([cv2.contourArea(hull) for hull in hulls])
            self.output_log.append(f"Minimum area: {min_area}\n")
            # re-plot the image in image_display
            self.image_display.replot_image_with_contours(image_to_visualize)

            # output the minmum perimeter of the contours
            self.output_log.append("----------- Image processed -----------\n")

    def start_point_selection(self):
        if self.image_display.image is not None:
            self.selection_state = "selecting_stripe_points"
            self.selected_points = []
            self.output_log.append("################################################")
            self.output_log.append("Please select three stripe points on the image.")
        else:
            self.output_log.append("Please load an image first.")

    # -----------------------
    # Signal management methods
    # -----------------------

    def _on_image_loaded(self):
        self.output_log.append("----------- Image loaded -----------\n")

    def _on_image_clicked(self, x, y):
        if self.selection_state == "selecting_stripe_points":
            bgr = self.image_display.get_pixel_value(x, y)
            if bgr is not None:
                self.selected_points.append(
                    {"type": "stripe", "position": (x, y), "bgr": bgr}
                )
                self.output_log.append(f"stripe point ({x}, {y}), BGR: {bgr}")
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
                self.output_log.append(f"background point ({x}, {y}), BGR: {bgr}")
                if (
                    len([p for p in self.selected_points if p["type"] == "background"])
                    == 3
                ):
                    self.selection_state = None
                    self.output_log.append("Point selection completed.")
                    self.output_log.append(
                        "################################################\n"
                    )
                    self.handle_selected_points()
            else:
                self.output_log.append("Invalid point selected.")

    def handle_selected_points(self):
        # Process the selected points as needed
        stripe_points = [p for p in self.selected_points if p["type"] == "stripe"]
        background_points = [
            p for p in self.selected_points if p["type"] == "background"
        ]

        # store the BGR values of the selected points
        self.stripes_vectors = [np.array(p["bgr"]) for p in stripe_points]
        self.background_vectors = [np.array(p["bgr"]) for p in background_points]
        self.target_background_vector = np.mean(self.background_vectors, axis=0)
