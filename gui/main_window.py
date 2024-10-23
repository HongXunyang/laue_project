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
    QCheckBox,
    QGridLayout,
)
from .image_display import ImageDisplay
from .matplotlib_canvas import MatplotlibCanvas
from .helper_functions import process_data
from PyQt5.QtCore import Qt
from utils import visualize_contours
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from close_packing import batch_optimization
from config.config import batch_optimization_kwargs
from utils import visualize_sampleholder


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
        self.default_batch_optimization_kwargs = batch_optimization_kwargs
        self.sampleholder = None  # sampleholder object

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
        self.epsilon_input.setPlaceholderText(" 2.5")
        self.lowercut_input.setPlaceholderText(" 100")
        self.area_lowercut_input.setPlaceholderText(" 2000")
        self.gaussian_size_input.setPlaceholderText(" 7")
        contour_finding_params_layout.addRow("Epsilon:", self.epsilon_input)
        contour_finding_params_layout.addRow("Lowercut:", self.lowercut_input)
        contour_finding_params_layout.addRow("Area lowercut:", self.area_lowercut_input)
        contour_finding_params_layout.addRow(
            "Gauss. filter size:", self.gaussian_size_input
        )

        contour_finding_params.setLayout(contour_finding_params_layout)

        # B-2: Close packing parameters
        # - create MAIN-widget
        # - create layout
        # - add SUB-widgets to layout
        # - set layout in MAIN-widget

        # parameters needed: number_system: int, step_size:float, number_of_iterations:int,temperature:float,contour_buffer_multiplier:float,optimize_shape:str,gravity_multiplier:float,

        # all the boolean parameters are set by a toggle button
        # all the integer parameters are set by a QLineEdit
        # all the float parameters are set by a QLineEdit as well and later on converted to float

        close_packing_params = QGroupBox("close_packing_params")
        close_packing_params_layout = QFormLayout()

        # Create QLineEdit widgets for integer and float parameters
        self.number_system_input = QLineEdit()
        self.number_system_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["number_system"])
        )
        self.step_size_input = QLineEdit()
        self.step_size_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["step_size"])
        )
        self.number_of_iterations_input = QLineEdit()
        self.number_of_iterations_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["number_of_iterations"])
        )
        self.temperature_input = QLineEdit()
        self.temperature_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["temperature"])
        )
        self.contour_buffer_multiplier_input = QLineEdit()
        self.contour_buffer_multiplier_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["contour_buffer_multiplier"])
        )
        self.optimize_shape_input = QLineEdit()
        self.optimize_shape_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["optimize_shape"])
        )
        self.gravity_multiplier_input = QLineEdit()
        self.gravity_multiplier_input.setPlaceholderText(
            "" + str(batch_optimization_kwargs["gravity_multiplier"])
        )
        # boolean parameters are set by a toggle button
        # is_gravity:bool,is_update_sampleholder:bool,# is_contour_buffer:bool,is_plot_area:bool

        # Create QPushButton widgets for boolean parameters
        self.is_gravity_button = QPushButton("Enable Gravity")
        self.is_gravity_button.setCheckable(True)
        self.is_gravity_button.setChecked(True)
        self.is_gravity_button.setObjectName("is_gravity_button")

        self.is_update_sampleholder_button = QPushButton("Update Sample Holder")
        self.is_update_sampleholder_button.setCheckable(True)
        self.is_update_sampleholder_button.setChecked(True)
        self.is_update_sampleholder_button.setObjectName(
            "is_update_sampleholder_button"
        )

        self.is_contour_buffer_button = QPushButton("Enable Contour Buffer")
        self.is_contour_buffer_button.setCheckable(True)
        self.is_contour_buffer_button.setChecked(True)
        self.is_contour_buffer_button.setObjectName("is_contour_buffer_button")

        self.is_plot_area_button = QPushButton("Plot Area")
        self.is_plot_area_button.setCheckable(True)
        self.is_plot_area_button.setChecked(False)
        self.is_plot_area_button.setObjectName("is_plot_area_button")

        close_packing_params_layout.addRow("No. of System:", self.number_system_input)
        close_packing_params_layout.addRow("Step Size:", self.step_size_input)
        close_packing_params_layout.addRow(
            "Number of Iterations:", self.number_of_iterations_input
        )
        close_packing_params_layout.addRow("Temperature:", self.temperature_input)
        close_packing_params_layout.addRow(
            "Contour Buffer Multiplier:", self.contour_buffer_multiplier_input
        )
        close_packing_params_layout.addRow("Optimize Shape:", self.optimize_shape_input)
        close_packing_params_layout.addRow(
            "Gravity Multiplier:", self.gravity_multiplier_input
        )
        close_packing_params_layout.addRow(self.is_gravity_button)
        close_packing_params_layout.addRow(self.is_update_sampleholder_button)
        close_packing_params_layout.addRow(self.is_contour_buffer_button)
        close_packing_params_layout.addRow(self.is_plot_area_button)

        close_packing_params.setLayout(close_packing_params_layout)

        # B-3: Buttons
        controls = QGroupBox("controls")
        controls_layout = QVBoxLayout()

        # select-points button
        self.select_points_button = QPushButton("Select Points")
        controls_layout.addWidget(self.select_points_button)
        self.select_points_button.setObjectName("select_points_button")

        # Process Button
        self.process_button = QPushButton("Process Image")
        controls_layout.addWidget(self.process_button)
        self.process_button.setObjectName("process_button")

        # Close Packing Button
        self.close_packing_button = QPushButton("Start Close Packing")
        controls_layout.addWidget(self.close_packing_button)
        self.close_packing_button.setObjectName("close_packing_button")

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
        self.close_packing_button.clicked.connect(self.close_packing)
        self.close_packing_button.clicked.connect(self.plot_close_packing_results)
        # -----------------------
        # Signal management
        # -----------------------
        self.image_display.image_loaded_signal.connect(self._on_image_loaded)
        self.image_display.point_clicked_signal.connect(self._on_image_clicked)

    def process_image(self):
        if self.image_display.image is not None:
            self.output_log.append("----------- 🏃‍ Start [Image Process] -----------\n")
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
            min_area = min([cv2.contourArea(hull) for hull in hulls])
            self.output_log.append(f"Minimum area: {min_area}\n")
            # re-plot the image in image_display
            self.image_display.replot_image_with_contours(image_to_visualize)

            # update the sampleholder object
            samples_list = generate_sample_objects(approximated_contours, hulls)
            self.sampleholder = generate_sampleholder_object(samples_list)
            self.output_log.append("----------- ✔️ End [Image Process] -----------\n")

    def start_point_selection(self):
        if self.image_display.image is not None:
            self.selection_state = "selecting_stripe_points"
            self.selected_points = []
            self.output_log.append("################################################")
            self.output_log.append("Please select three stripe points on the image.")
        else:
            self.output_log.append("Please load an image first.")

    def close_packing(self):
        """
        start close packing process and display the results

        - read close packing keyword arguments
        - run the close packing algorithm
        """
        self.output_log.append("----------- 🏃‍ Start [close packing] -----------\n")
        local_batch_optimization_kwargs = self.get_local_batch_optimization_kwargs()
        optimized_configuration_list, area_list, sorted_indices = batch_optimization(
            self.sampleholder,
            **local_batch_optimization_kwargs,
        )
        self.output_log.append("----------- ✔️ End of [Close Packing] -----------\n")

    def plot_close_packing_results(self):
        """plot this on the matplotlib canvas"""
        # clear the canvas
        self.matplotlib_canvas.axes.clear()
        visualize_sampleholder(self.sampleholder, self.matplotlib_canvas.axes)
        self.matplotlib_canvas.axes.set(xticks=[], yticks=[])
        self.matplotlib_canvas.draw()

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

    # -----------------------
    # Helper methods
    # -----------------------
    def get_local_batch_optimization_kwargs(self):
        # Retrieve and convert parameters
        number_system = (
            int(self.number_system_input.text())
            if self.number_system_input.text()
            else self.default_batch_optimization_kwargs["number_system"]
        )
        step_size = (
            float(self.step_size_input.text())
            if self.step_size_input.text()
            else self.default_batch_optimization_kwargs["step_size"]
        )
        number_of_iterations = (
            int(self.number_of_iterations_input.text())
            if self.number_of_iterations_input.text()
            else self.default_batch_optimization_kwargs["number_of_iterations"]
        )
        temperature = (
            float(self.temperature_input.text())
            if self.temperature_input.text()
            else self.default_batch_optimization_kwargs["temperature"]
        )
        contour_buffer_multiplier = (
            float(self.contour_buffer_multiplier_input.text())
            if self.contour_buffer_multiplier_input.text()
            else self.default_batch_optimization_kwargs["contour_buffer_multiplier"]
        )
        optimize_shape = (
            self.optimize_shape_input.text()
            if self.optimize_shape_input.text()
            else self.default_batch_optimization_kwargs["optimize_shape"]
        )
        gravity_multiplier = (
            float(self.gravity_multiplier_input.text())
            if self.gravity_multiplier_input.text()
            else self.default_batch_optimization_kwargs["gravity_multiplier"]
        )

        # Retrieve boolean parameters
        is_gravity = self.is_gravity_button.isChecked()
        is_update_sampleholder = self.is_update_sampleholder_button.isChecked()
        is_contour_buffer = self.is_contour_buffer_button.isChecked()
        is_plot_area = self.is_plot_area_button.isChecked()

        # Now you can use these parameters as needed
        local_batch_optimization_kwargs = {
            "number_system": number_system,
            "step_size": step_size,
            "number_of_iterations": number_of_iterations,
            "temperature": temperature,
            "contour_buffer_multiplier": contour_buffer_multiplier,
            "optimize_shape": optimize_shape,
            "gravity_multiplier": gravity_multiplier,
            "is_gravity": is_gravity,
            "is_update_sampleholder": is_update_sampleholder,
            "is_contour_buffer": is_contour_buffer,
            "is_plot_area": is_plot_area,
            "is_plot": False,
            "is_print": False,
        }

        return local_batch_optimization_kwargs
