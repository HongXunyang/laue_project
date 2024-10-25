import numpy as np

image2contours_kwargs = dict(
    epsilon=2.5,
    lowercut=100,
    area_lowercut=1000,
    threshold=100,
    gaussian_window=np.array([5, 5]),
    is_gaussian_filter=True,
)

physical_size = dict(
    sampleholder_thickness=60,
    sampleholder_size=np.array([1000, 1000]),
)

batch_optimization_kwargs = dict(
    number_system=5,
    is_plot=False,
    is_print=True,
    step_size=20,
    number_of_iterations=4000,
    temperature=300,
    contour_buffer_multiplier=1.01,
    optimize_shape="min_circle",
    is_gravity=True,
    gravity_multiplier=0.7,
    gravity_off_at=2300,
    is_update_sampleholder=True,
    is_contour_buffer=True,
    is_plot_area=True,
    is_rearrange_vertices=True,
)

config = dict(
    data_path="../data/",
    sampleholder_dict_filename="sampleholder.json",
    sampleholder_cad_filename="engraved_sampleholder.stl",
    temp_images_path="../temp_images/",
)

plot_area_evolution_kwargs = dict(color="dodgerblue", alpha=0.5, linewidth=1.5)

plot_ratio_evolution_kwargs = dict(color="darkorange", alpha=0.5, linewidth=1.5)
