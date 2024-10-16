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

# finding contours and hulls
contours, approximated_contours, hulls = image2contours(
    image,
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    is_gaussian_filter=True,
)

# visualize contours
image_to_visualize = image_to_visualize = visualize_contours(
    image,
    approximated_contours,
    hulls,
)

# create samples objects and sample holder object
samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)
print(sampleholder)
sampleholder.print_samples()
cv2.waitKey(0)

if True:
    cv2.imwrite("../images/fake_holder_with_samples_contours.jpg", image_to_visualize)
