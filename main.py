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
    visualize_contours,
    save_sampleholder,
    animate_config_evolution,
)
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from close_packing import optimization, visualize_vertices_list, batch_optimization
from config.config import physical_size, batch_optimization_kwargs, config


start_time = time.time()
# ----------- image pre=processing ----------- #
stripes_vectors = [
    np.array([95, 86, 167]),
    np.array([57, 48, 139]),
    np.array([72, 66, 137]),
]
target_background_vector = np.array([202, 209, 206])
background_vectors = [
    np.array([202, 209, 206]),
    np.array([190, 201, 199]),
    np.array([182, 185, 183]),
]
# Load image
image = cv2.imread("../images/fake_holder_with_samples.jpg")
rows, columns, channels = image.shape

# crop image
image = image[
    int(0.15 * rows) : int(0.435 * rows), int(0.1 * columns) : int(0.9 * columns)
]
# compress image
image = cv2.resize(image, (rows // 4, columns // 4), interpolation=cv2.INTER_AREA)
rows, columns, channels = image.shape
# ----------- end of image pre-processing ----------- #

# ----------- contour finding ----------- #
contours, approximated_contours, hulls, _ = image2contours(
    image,
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    epsilon=2.5,
    lowercut=100,
    area_lowercut=2000,
    gaussian_window=(5, 5),
    is_gaussian_filter=True,
    threshold=150,
)

# visualize contours
image_to_visualize = visualize_contours(
    image, approximated_contours, hulls, is_plot=False
)
end_time = time.time()
print(f"image processed time: {end_time - start_time} seconds\n")
# cv2.waitKey(0)
# create samples objects and sample holder object
samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)
# ----------- end of contour finding ----------- #

# ----------- optimization ----------- #
if True:
    start_time = time.time()
    optimized_configuration_list, area_list, sorted_indices, _ = batch_optimization(
        sampleholder,
        **batch_optimization_kwargs,
    )
    end_time = time.time()

    print(f"optimization time: {end_time - start_time} seconds\n")
    fig, ax = plt.subplots()
    visualize_sampleholder(
        sampleholder,
        ax=ax,
        is_plot_contour=False,
        is_plot_hull=True,
    )

# ----------- end of optimization ----------- #


# ----------- convert Samples to CAD ----------- #
vertices_list = optimized_configuration_list[sorted_indices[0]]
vertices_list_to_cad(
    vertices_list,
)
# ----------- end  ----------- #


# ----------- convert Sample holder to CAD ----------- #
# adjust the size of the sample holder
sampleholder.size = physical_size["sampleholder_size"]
sampleholder.shape = "circle"
sampleholder.thickness = physical_size["sampleholder_thickness"]
# get the minimum enclosing circle
sampleholder.update_min_circle()
sampleholder.radius = sampleholder.radius * 1.1

sampleholder_to_cad(
    sampleholder,
)

# ----------- end  ----------- #

plt.show()