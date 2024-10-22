import numpy as np

image2contours_kwargs = dict(
    epsilon=2.5, lowercut=100, area_lowercut=1000, gaussian_window=np.array([7, 7])
)

physical_size = dict(
    sampleholder_thickness=60,
    sampleholder_size=np.array([1000, 1000]),
)

batch_optimization_kwargs = dict(
    number_system=13,
    is_plot=True,
    is_print=True,
    step_size=10,
    number_of_iteration=20000,
    temperature=1500,
    contour_buffer_multiplier=1.05,
    optimize_shape="min_circle",
    is_gravity=True,
    is_update_sampleholder=True,
    is_contour_buffer=True,
    is_plot_area=True,
)
