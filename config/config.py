""" 
__version__ = 1.0
__author__ = "Xunyang Hong"

updated on 26th Oct. 2024
"""

import numpy as np

image2contours_kwargs = dict(
    epsilon=2.5,
    lowercut=30,
    area_lowercut=300,
    threshold=50,
    gaussian_window=np.array([5, 5]),
    is_gaussian_filter=False,
)

physical_size = dict(
    sampleholder_thickness=1,  # in the unit of mm
    sampleholder_size=np.array([1000, 1000]),
    sample_thickness=1.05,  # in the unit of mm
    sampleholder_radius_multiplier=1.2,  # if set to 1.2, it means the radius of the sample holder is 20% bigger than the min-enclosing circle of all samples.
)

batch_optimization_kwargs = dict(
    number_system=3,
    is_print=True,
    step_size=20,
    number_of_iterations=10000,
    temperature=300,
    contour_buffer_multiplier=1.01,
    optimize_shape="min_circle",
    is_gravity=True,
    gravity_multiplier=0.7,
    gravity_off_at=2300,
    is_contour_buffer=True,
    is_rearrange_vertices=True,
    is_save_results=True,
)

config = dict(
    temporary_output_folder="temporary_output/",
    data_path="temporary_output/",
    sampleholder_dict_filename="sampleholder.json",
    sampleholder_cad_filename="engraved_sampleholder.stl",
    samples_cad_filename="samples.stl",
)

plot_area_evolution_kwargs = dict(color="dodgerblue", alpha=0.5, linewidth=1.5)

plot_ratio_evolution_kwargs = dict(color="darkorange", alpha=0.5, linewidth=1.5)

tests_config = dict(
    test_image_path="tests/test_data/test_image.jpg",
    stripes_vectors=[
        np.array([119, 119, 119]),
        np.array([100, 100, 100]),
        np.array([120, 120, 120]),
    ],
    target_background_vector=np.array([209, 209, 209]),
    background_vectors=[
        np.array([209, 209, 209]),
        np.array([190, 190, 190]),
        np.array([220, 220, 220]),
    ],
)
