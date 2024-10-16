import cv2
import numpy as np
import matplotlib.pyplot as plt
from packages import (
    unify_background,
    remove_stripes,
    contours2polygons,
    image2contours,
    visualize_contours,
    generate_contour_object,
    generate_sample_object,
    generate_sample_objects,
    generate_sampleholder_object,
)

# pre-defined parameters
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
contours_kwargs = {
    "color": (0, 0, 255),
    "thickness": 4,
}
hulls_kwargs = {
    "color": (0, 255, 0),
    "thickness": 3,
}

# Load image
image = cv2.imread("../images/fake_holder_with_samples.jpg")
rows, columns, channels = image.shape
image = image[
    int(0.15 * rows) : int(0.435 * rows), int(0.1 * columns) : int(0.9 * columns)
]
image = cv2.resize(image, (rows // 4, columns // 4), interpolation=cv2.INTER_AREA)
rows, columns, channels = image.shape
contours, approximated_contours, hulls = image2contours(
    image,
    gaussian_window=(7, 7),
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    epsilon=2.5,
    lowercut=100,
)
image_to_visualize = image_to_visualize = visualize_contours(
    image,
    approximated_contours,
    hulls,
    contours_kwargs=contours_kwargs,
    hulls_kwargs=hulls_kwargs,
)

samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)
print(sampleholder)
sampleholder.print_samples()
cv2.waitKey(0)
