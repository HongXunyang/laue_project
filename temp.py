from close_packing import optimization, visualize_vertices_list, batch_optimization
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time, cv2
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
    remove_stripes,
    unify_background,
)
from utils import (
    visualize_sampleholder,
    visualize_contours,
    save_sampleholder,
    visualize_sampleholder_dict,
)

start_time = time.time()
# ----------- image pre=processing ----------- #
print(" \n---------- image pre-processing ----------\n")
stripes_vectors = [
    np.array([119, 119, 119]),
    np.array([119, 119, 119]),
    np.array([119, 119, 119]),
]
target_background_vector = np.array([209, 209, 209])
background_vectors = [
    np.array([209, 209, 209]),
    np.array([209, 209, 209]),
    np.array([209, 209, 209]),
]
# Load image
image = cv2.imread("../images/sissi_circle_sample.jpg")
rows, columns, channels = image.shape
# ----------- end of image pre-processing ----------- #

# ----------- contour finding ----------- #

image_stripes_removed = remove_stripes(image, stripes_vectors, target_background_vector)
# plot this image
image_unified = unify_background(
    image_stripes_removed, background_vectors, target_background_vector
)

contours, approximated_contours, hulls = image2contours(
    image,
    is_preprocess=True,
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    is_gaussian_filter=False,
    gaussian_window=(3, 3),
)
