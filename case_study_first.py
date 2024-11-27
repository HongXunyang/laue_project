from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time, cv2
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import (
    visualize_sampleholder,
    rearrange_samples_indeces,
    visualize_contours,
    animate_config_evolution,
)
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from config.config import (
    batch_optimization_kwargs,
    tests_config,
    image2contours_kwargs,
)
from close_packing import batch_optimization, optimization


mm_per_pixel = 5.0 / (1450.0 - 1282.0)


# ----------- image pre-processing ----------- #

start_time = time.time()
stripes_vectors = np.array([[124, 128, 147], [105, 115, 133], [104, 115, 142]])
background_vectors = np.array([[175, 169, 162], [184, 178, 171], [189, 181, 174]])
target_background_vector = np.array([180, 175, 168])

# Load image
image = cv2.imread("../images/samples_from_jasmin.jpg")
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
    is_output_image=True,
)


end_time = time.time()
print(f"image processed time: {end_time - start_time} seconds\n")

# create samples objects and sample holder object
samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)
contours = [sample.contour_original.contour for sample in sampleholder.samples_list]
hulls = [sample.contour_original.hull for sample in sampleholder.samples_list]


# ----------- end of contour finding ----------- #
fig, axs = plt.subplots(2, 1)
visualize_sampleholder(sampleholder, is_min_circle=False, ax=axs[0])


# time for rotation
phi_list = [
    27,
    0,
    -24,
    -23,
    8,
    37,
    38,
    -15,
    -25,
    25,
    26,
    -43,
    -10,
    -44,
    -6,
    15,
    -15,
    40,
    30,
    -40,
    -13,
    0,
    -31,
]
for i, phi in enumerate(phi_list):
    sample = sampleholder.samples_list[i]
    sample.phi_offset = phi
    sample.reorient()
visualize_sampleholder(sampleholder, is_min_circle=False, ax=axs[1])
# ax 0 title is original, ax 1 title is reoriented
axs[0].set_title("original")
axs[1].set_title("reoriented")
fig.tight_layout()
fig.savefig("temporary_output/reoriented_samples.jpg")

if False:
    # generate animation to determine the best parameters for the contour finding
    start_time = time.time()
    best_vertices_list, best_area, optimization_history = optimization(
        sampleholder,
        number_of_iterations=20000,
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
    visualize_sampleholder(sampleholder, is_min_circle=True)

if True:
    start_time = time.time()
    best_configuration, area_list, sorted_indices, _ = batch_optimization(
        sampleholder,
        number_system=10,
        is_print=True,
        step_size=20,
        number_of_iterations=30000,
        temperature=100,
        contour_buffer_multiplier=1.05,
        optimize_shape="min_circle",
        is_gravity=True,
        gravity_multiplier=0.25,
        gravity_off_at=2000,
        is_contour_buffer=True,
        is_rearrange_vertices=True,
        is_save_results=True,
    )
    end_time = time.time()
    print(f"optimization time: {end_time - start_time} seconds\n")

    visualize_sampleholder(sampleholder, is_min_circle=False)
    # ----------- end of optimization ----------- #
    # # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # ----------- convert Samples to CAD --------- #

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
        mm_per_pixel=mm_per_pixel,
    )
    # ----------- end  ----------- #
plt.show()
print("done")
