import numpy as np
import cv2, json, time, os
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
from PyQt5.QtCore import QThread
from utils import visualize_contours
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from PyQt5.QtGui import QTextCursor
from close_packing import batch_optimization
from config.config import batch_optimization_kwargs, config, image2contours_kwargs
from utils import visualize_sampleholder, visualize_area_evolution, save_sampleholder
from .worker import ClosePackingWorker
from to_cad import sampleholder_to_cad


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
        self.area_evolution_list = None
        self.default_image2contours_kwargs = image2contours_kwargs
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
        self.threshold_input = QLineEdit()
        # Set placeholders or default values
        self.epsilon_input.setPlaceholderText(str(image2contours_kwargs["epsilon"]))
        self.lowercut_input.setPlaceholderText(str(image2contours_kwargs["lowercut"]))
        self.area_lowercut_input.setPlaceholderText(
            str(image2contours_kwargs["area_lowercut"])
        )
        self.gaussian_size_input.setPlaceholderText(
            str(image2contours_kwargs["gaussian_window"][0]) + ", set to 0 if no filter"
        )
        self.threshold_input.setPlaceholderText(str(image2contours_kwargs["threshold"]))
        contour_finding_params_layout.addRow("Epsilon:", self.epsilon_input)
        contour_finding_params_layout.addRow("Lowercut:", self.lowercut_input)
        contour_finding_params_layout.addRow("Area lowercut:", self.area_lowercut_input)
        contour_finding_params_layout.addRow(
            "Gauss. filter size:", self.gaussian_size_input
        )
        contour_finding_params_layout.addRow("Threshold:", self.threshold_input)

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
        self.gravity_off_at_input = QLineEdit()
        self.gravity_off_at_input.setPlaceholderText(
            str(batch_optimization_kwargs["gravity_off_at"])
        )
        # boolean parameters are set by a toggle button
        # is_gravity:bool,is_update_sampleholder:bool,# is_contour_buffer:bool, is_save_results:bool

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

        self.is_save_results_button = QPushButton("Save Results")
        self.is_save_results_button.setCheckable(True)
        self.is_save_results_button.setChecked(True)
        self.is_save_results_button.setObjectName("is_save_results_button")

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
        close_packing_params_layout.addRow(
            "Turn off gravity at step:", self.gravity_off_at_input
        )

        close_packing_params_layout.addRow(self.is_gravity_button)
        close_packing_params_layout.addRow(self.is_update_sampleholder_button)
        close_packing_params_layout.addRow(self.is_contour_buffer_button)
        close_packing_params_layout.addRow(self.is_save_results_button)

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

        # Convert to CAD Button
        self.convert_to_cad_button = QPushButton("Convert to CAD")
        controls_layout.addWidget(self.convert_to_cad_button)
        self.convert_to_cad_button.setObjectName("convert_to_cad_button")

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
        self.convert_to_cad_button.clicked.connect(self.convert_to_cad)
        # -----------------------
        # Signal management
        # -----------------------
        self.image_display.image_loaded_signal.connect(self._on_image_loaded)
        self.image_display.point_clicked_signal.connect(self._on_image_clicked)

    def process_image(self):
        if self.image_display.image is not None:
            self.output_log.append("----------- 🏃‍ Start [Image Process] -----------")
            # Retrieve parameters, use defaults if input is empty
            image2contours_kwargs = self.get_local_image2contours_kwargs()
            # Load the image path
            image_path = self.image_display.image_path
            image = self.image_display.image
            rows, columns, channels = image.shape
            number_pixels = rows * columns
            estimated_time = int(number_pixels / (1024 * 2048) * 0.25)
            self.output_log.append(f"Estimated time: {estimated_time} seconds")
            cv2.waitKey(0)
            _, approximated_contours, hulls, logging_dict = image2contours(
                image,
                stripes_vectors=self.stripes_vectors,
                background_vectors=self.background_vectors,
                **image2contours_kwargs,
            )
            # update the sampleholder object
            samples_list = generate_sample_objects(approximated_contours, hulls)
            self.sampleholder = generate_sampleholder_object(
                samples_list, is_rearrange_indeces=True
            )
            samples_list = self.sampleholder.samples_list
            rearranged_contours = [
                sample.contour_original.contour
                for sample in self.sampleholder.samples_list
            ]
            rearranged_hulls = [
                sample.contour_original.hull
                for sample in self.sampleholder.samples_list
            ]
            image_to_visualize = visualize_contours(
                image,
                rearranged_contours,
                rearranged_hulls,
                is_plot=False,
                is_output_image=True,
            )

            min_area = min([cv2.contourArea(hull) for hull in hulls])
            max_area = max([cv2.contourArea(hull) for hull in hulls])
            self.output_log.append(
                f"Minimum area: {min_area}, maximum area: {max_area}\n"
            )
            # re-plot the image in image_display
            self.image_display.replot_image_with_contours(image_to_visualize)

            # logging
            self.output_log.append(f"Information about the Image:")
            # max brighness, min brightness, width in pixel, height in pixel
            self.output_log.append(f"max brightness: {logging_dict['max_brightness']}")
            self.output_log.append(
                f"current threshold: {image2contours_kwargs['threshold']}"
            )
            self.output_log.append(f"min brightness: {logging_dict['min_brightness']}")
            self.output_log.append(f"width in pixel: {logging_dict['width']}")
            self.output_log.append(f"height in pixel: {logging_dict['height']}")

            self.output_log.append("----------- ✔️ End [Image Process] -----------\n")

    def start_point_selection(self):
        if self.image_display.image is not None:
            self.selection_state = "selecting_stripe_points"
            self.selected_points = []
            self.output_log.append("----------- 🏃‍ Start [Point Selection] -----------")
            self.output_log.append("Please select three stripe points on the image.")
        else:
            self.output_log.append("Please load an image first.")

    def close_packing(self):
        """
        start close packing process and display the results

        - read close packing keyword arguments
        - run the close packing algorithm
        """
        self.output_log.append("----------- 🏃‍ Start [close packing] -----------")

        # get local batch optimization kwargs
        local_batch_optimization_kwargs = self.get_local_batch_optimization_kwargs()

        # create a worker thread
        self.thread = QThread()

        # Create a worker object
        self.worker = ClosePackingWorker(
            self.sampleholder, local_batch_optimization_kwargs
        )

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.close_packing_finished)
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread
        self.thread.start()

        self.output_log.append("Close packing started in background thread.")

    def update_progress(self, progress, estimated_total_time, remaining_time):
        estimated_total_time_str = time.strftime(
            "%H:%M:%S", time.gmtime(estimated_total_time)
        )
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

        if remaining_time < 0:
            remaining_time = 0
        if progress <= 1:
            self.output_log.append(
                f"Progress: {progress:.1f}% | Estimating total time..."
            )
        else:
            self.output_log.append(
                f"Progress: {progress:.1f}% | Total Time: {estimated_total_time_str} | Remaining: {remaining_time_str}"
            )
        # Scroll to the end
        self.output_log.moveCursor(QTextCursor.End)

    def close_packing_finished(self, sampleholder, area_evolution_list):
        """
        Handle the completion of the close packing process.
        """
        # Update the sampleholder and area_evolution_list
        self.sampleholder = sampleholder
        self.area_evolution_list = area_evolution_list

        # Save the results
        save_sampleholder(
            self.sampleholder, config["data_path"], config["sampleholder_dict_filename"]
        )

        self.output_log.append("----------- ✔️ End of [Close Packing] -----------\n")

        # Plot the results
        self.plot_close_packing_results()

    def plot_close_packing_results(self):
        """plot this on the matplotlib canvas"""
        # clear out current canvas image
        self.matplotlib_canvas.ax_sampleholder.clear()
        self.matplotlib_canvas.ax_evolution.clear()

        # get kwargs
        local_batch_optimization_kwargs = self.get_local_batch_optimization_kwargs()

        # plot the sampleholder
        self.matplotlib_canvas.ax_sampleholder.clear()
        self.sampleholder.update()
        visualize_sampleholder(
            self.sampleholder, self.matplotlib_canvas.ax_sampleholder
        )
        self.matplotlib_canvas.ax_sampleholder.set(xticks=[], yticks=[])

        # plot the area evolution
        self.matplotlib_canvas.ax_evolution.clear()
        ax_ratio = self.matplotlib_canvas.ax_evolution.twinx()
        visualize_area_evolution(
            self.sampleholder,
            self.area_evolution_list,
            self.matplotlib_canvas.ax_evolution,
            ax_ratio,
        )

        self.matplotlib_canvas.draw()

    def convert_to_cad(self):
        folder_path = config["temporary_output_folder"]
        filename = config["sampleholder_cad_filename"]
        sampleholder_to_cad(
            self.sampleholder,
            folder_path=folder_path,
            filename=filename,
        )
        self.output_log.append(
            f"Exported the sample holder to {folder_path} in STL format."
        )

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
                        "----------- ✔️ End [Point Selection] -----------\n"
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
        gravity_off_at = (
            int(self.gravity_off_at_input.text())
            if self.gravity_off_at_input.text()
            else self.default_batch_optimization_kwargs["gravity_off_at"]
        )
        # Retrieve boolean parameters
        is_gravity = self.is_gravity_button.isChecked()
        is_contour_buffer = self.is_contour_buffer_button.isChecked()
        is_save_results = self.is_save_results_button.isChecked()

        # Now you can use these parameters as needed
        local_batch_optimization_kwargs = {
            "number_system": number_system,
            "step_size": step_size,
            "number_of_iterations": number_of_iterations,
            "temperature": temperature,
            "contour_buffer_multiplier": contour_buffer_multiplier,
            "optimize_shape": optimize_shape,
            "gravity_multiplier": gravity_multiplier,
            "gravity_off_at": gravity_off_at,
            "is_gravity": is_gravity,
            "is_contour_buffer": is_contour_buffer,
            "is_save_results": is_save_results,
            "is_print": False,
        }

        return local_batch_optimization_kwargs

    def get_local_image2contours_kwargs(self):
        epsilon = (
            float(self.epsilon_input.text())
            if self.epsilon_input.text()
            else self.default_image2contours_kwargs["epsilon"]
        )
        lowercut = (
            int(self.lowercut_input.text())
            if self.lowercut_input.text()
            else self.default_image2contours_kwargs["lowercut"]
        )
        area_lowercut = (
            int(self.area_lowercut_input.text())
            if self.area_lowercut_input.text()
            else self.default_image2contours_kwargs["area_lowercut"]
        )
        threshold = (
            int(self.threshold_input.text())
            if self.threshold_input.text()
            else self.default_image2contours_kwargs["threshold"]
        )
        gaussian_window = (
            np.array(
                [
                    int(self.gaussian_size_input.text()),
                    int(self.gaussian_size_input.text()),
                ]
            )
            if self.gaussian_size_input.text()
            else self.default_image2contours_kwargs["gaussian_window"]
        )
        is_gaussian_filter = False if self.gaussian_size_input.text() == "0" else True

        image2contours_kwargs = dict(
            epsilon=epsilon,
            lowercut=lowercut,
            area_lowercut=area_lowercut,
            threshold=threshold,
            gaussian_window=gaussian_window,
            is_gaussian_filter=is_gaussian_filter,
        )
        return image2contours_kwargs
