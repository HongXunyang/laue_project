import numpy as np

image2contours_kwargs = dict(
    epsilon=2.5, lowercut=100, area_lowercut=1000, gaussian_window=np.array([7, 7])
)

physical_size = dict(
    sampleholder_thickness=60,
    sampleholder_size=np.array([1000, 1000]),
)

batch_optimization_kwargs = dict(
    number_system=300,
    is_plot=True,
    is_print=True,
    step_size=10,
    number_of_iterations=40000,
    temperature=700,
    contour_buffer_multiplier=1.01,
    optimize_shape="min_circle",
    is_gravity=True,
    gravity_multiplier=0.3,
    is_update_sampleholder=True,
    is_contour_buffer=True,
    is_plot_area=True,
)
