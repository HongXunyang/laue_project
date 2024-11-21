""" 
__version__ = 1.0
__author__ = "Xunyang Hong"

updated on 26th Oct. 2024

----------------------------------------------
# [Trial Run] Usage: 

Before doing any real application, users are suggested to go for a "trial run" to get a better understanding of the program. In the trial run, the image to process is a classic example. This image allows for a easy and quick contour finding. In this trial run, the users do not need to worry about the contour finding. The key point of this run is to figure out the best parameters for the optimization process.

The dictionary `STEP_CONTROL` controls the steps of the program. The following steps are the recommended steps: 

1. Run the scripts with `STEP_CONTROL["test"] = True`. This will generate an animation of the close packing process. Look closely at the optimization process and determine the best parameters for the contour finding. Users are encouraged to adjust these parameters for the `optimization` function: 
    - `number_of_iterations`
    - `step_size`
    - `temperature`
    - `gravity_multiplier`
    - `gravity_off_at`

2. Once you find the approprite parameters:
    - set `STEP_CONTROL["test"] = False`, and everything else to `True`. When this set to `True`, a more serious optimization process will be launched, i.e. `batch_optimization` where multiple systems will be involved and get optimized in parallel. This will help to find a better configuratio for the system. 
    - modify the corresponding parameters in `batch_optimization_kwargs` in `config/config.py`. Type in the appropriate parameters you found in the first step. They will be the parameters used for the bacth close packing process. 
    - additional parameters for `batch_optimization` are expected to modify as well: `number_system`: please type in a int number in [3, 1000]. 
    - if not sure about what the other keywords arguements are, please them as default.

3. Run the script again, the results will be plotted and saved in the `temporary_output/` folder. The output files include: 
    - different stages of the processed images: With this images, the user can check whether the parameters for the image processing is appropriate. IF NOT, modify the `image2contours_kwargs` in `config/config.py` and redo the process
    - The figure of the optimized configuration
    - the JSON file of the sample holder, containing information of the optimized configuration, the ratio, and etc.
    - the CAD/STL file of the engraved sample holder
    - the CAD/STL file of the samples configuration

----------------------------------------------
# [Real-life Application] Usage:
For any realy-life application, the users are suggested to supervise the whole process from the beginning, i.e. the image processing. 

All the parameters can be found and adjusted in the `config/config.py` file. It's recommended to adjust it in the config file for a better overall global controll of the program

Please follow the steps below:

1. Image processing: Set `STEP-CONTROL["contour_finding"]= True` and EVERYTHING ELSE to `False`. Please refer to the README.md file for the detailed information of the image processing. Long story short: the processing involves: manual selection of stripes color vectors, and background color vectors for the program to filter out stripes and background. The users need to adjust the following parameters:
    - `stripes_vectors`
    - `background_vectors`
    - `target_background_vector`
    - `epsilon`
    - `lowercut`
    - `area_lowercut`
    - `threshold`
    - `gaussian_window`
    - `is_gaussian_filter`
Check the `temporary_output/` folder for the processed images. If the contours are not found as expected, please adjust the parameters and redo the process.

2. Test Optimization: Once you obtain a satisfactory image processing result, set `STEP_CONTROL["test"]=True` to start a test run. This is again to determine the best parameters for the optimization process. Everything else is the same as the trial run.

3. Batch Optimization: Once you find the best parameters for the optimization process, set `STEP_CONTROL["test"]=False` and `STEP_CONTROL["close_packing"]=True` and `STEP_CONTROL["convert_to_cad"]=True`. Everything else is the same as the trial run. The results will be saved in the `temporary_output/` folder.
"""

import time, cv2
from matplotlib import pyplot as plt
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import animate_config_evolution
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from config.config import (
    batch_optimization_kwargs,
    tests_config,
    image2contours_kwargs,
)
from close_packing import batch_optimization, optimization


STEP_CONTROL = dict(
    test=True, contour_finding=True, close_packing=True, convert_to_cad=True
)


# ----------- image pre-processing ----------- #
if STEP_CONTROL["contour_finding"] or STEP_CONTROL["test"]:
    start_time = time.time()
    stripes_vectors, background_vectors, target_background_vector = (
        tests_config["stripes_vectors"],
        tests_config["background_vectors"],
        tests_config["target_background_vector"],
    )
    # Load image
    image = cv2.imread(tests_config["test_image_path"])
    rows, columns, channels = image.shape

    # ----------- end of image pre-processing ----------- #
    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # ---------------- contour finding ------------------ #

    # Find the contours of the samples in the image
    # by default, the output images will be saved in the `temporary_output/` folder.
    contours, approximated_contours, hulls, _ = image2contours(
        image,
        stripes_vectors=stripes_vectors,
        background_vectors=background_vectors,
        **image2contours_kwargs,
    )
    end_time = time.time()
    print(f"image processed time: {end_time - start_time} seconds\n")

    # create samples objects and sample holder object
    samples_list = generate_sample_objects(approximated_contours, hulls)
    sampleholder = generate_sampleholder_object(samples_list)
# ----------- end of contour finding ----------- #
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------------- Test ----------------- #
if STEP_CONTROL["test"]:
    # generate animation to determine the best parameters for the contour finding
    start_time = time.time()
    best_vertices_list, best_area, optimization_history = optimization(
        sampleholder,
        number_of_iterations=35000,
        step_size=25,
        temperature=400,
        gravity_multiplier=0.5,
        gravity_off_at=2700,
        contour_buffer_multiplier=1.01,
        optimize_shape="min_circle",
        is_rearrange_vertices=True,
        is_gravity=True,
        is_update_sampleholder=True,
        is_contour_buffer=True,
        is_plot_evolution=False,
        is_record_area_history=True,
        is_record_configuration_history=True,
    )
    end_time = time.time()
    sampleholder.update()
    print(f"optimization time: {end_time - start_time} seconds\n")
    fig_ani, axs = plt.subplots(1, 2, figsize=(8, 4))
    configurations = optimization_history["vertices_list_evolution"]
    area_evolution = optimization_history["area_evolution"]
    animate_config_evolution(
        configurations,
        area_evolution,
        samples_area=sampleholder.samples_area,
        fig=fig_ani,
        axs=axs,
        is_save=True,
        filename="test_animation.mp4",
        max_duration=10,
    )
# ------------------- end of test -------------- #
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------------- Optimization --------------- #
if STEP_CONTROL["close_packing"] and not STEP_CONTROL["test"]:
    start_time = time.time()
    best_configuration, area_list, sorted_indices, _ = batch_optimization(
        sampleholder,
        **batch_optimization_kwargs,
    )
    end_time = time.time()
    print(f"optimization time: {end_time - start_time} seconds\n")

# ----------- end of optimization ----------- #
# # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------- convert Samples to CAD --------- #
if STEP_CONTROL["convert_to_cad"] and not STEP_CONTROL["test"]:
    vertices_list = sampleholder.vertices_list
    # Save the configuration of the samples to a CAD (STL) file
    # by default, the output CAD files will be saved in the `temporary_output/` folder.
    vertices_list_to_cad(
        vertices_list,
    )
    # update sampleholder object
    sampleholder.shape = "circle"
    sampleholder.update()

    # Save the engraved sampleholder to a CAD (STL) file
    # by default, the output CAD files will be saved in the `temporary_output/` folder.
    sampleholder_to_cad(
        sampleholder,
    )
# ----------- end  ----------- #
plt.show()
