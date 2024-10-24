from close_packing import optimization, visualize_vertices_list, batch_optimization
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
    visualize_sampleholder_dict,
)

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
contours, approximated_contours, hulls = image2contours(
    image,
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    epsilon=2.5,
    lowercut=100,
    area_lowercut=2000,
    gaussian_window=(5, 5),
    is_gaussian_filter=True,
    isprint=False,
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


optimization(
    sampleholder,
    step_size=10,
    number_of_iterations=1000,
    temperature=500,
    contour_buffer_multiplier=1.05,
    optimize_shape="min_circle",
    is_gravity=False,
    is_update_sampleholder=True,
    is_contour_buffer=True,
)

folder_path = "../data/"
filename = "sampleholder.json"
sampleholder_dict = save_sampleholder(sampleholder, folder_path, filename)

visualize_sampleholder_dict(sampleholder_dict)
plt.show()
