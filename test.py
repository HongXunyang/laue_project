import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
from classes import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
    visualize_sampleholder,
    visualize_contours,
)


start_time = time.time()
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


# Load image
image = cv2.imread("../images/fake_holder_with_samples.jpg")
rows, columns, channels = image.shape

# crop image
# image = image[
#    int(0.15 * rows) : int(0.435 * rows), int(0.1 * columns) : int(0.9 * columns)
# ]
# compress image
# image = cv2.resize(image, (rows // 4, columns // 4), interpolation=cv2.INTER_AREA)
rows, columns, channels = image.shape

# finding contours and hulls


def your_function():
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
    return contours, approximated_contours, hulls
