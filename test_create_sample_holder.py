import cv2
import numpy as np
import matplotlib.pyplot as plt
from packages import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
    visualize_sampleholder,
    visualize_contours,
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
image_to_visualize = visualize_contours(
    image,
    approximated_contours,
    hulls,
)
cv2.waitKey(0)
# create samples objects and sample holder object
samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)


# assign random phi_offset to samples
for sample in sampleholder.samples_list:
    sample.phi_offset = np.random.uniform(0, 180)
    sample.reorient()

# assign random position offset to the samples
for sample in sampleholder.samples_list:
    position_offset = np.random.uniform(-100, 100, 2)
    sample.position_new = sample.position_original + position_offset
    sample.relocate()

# visualize the sample holder
fig, ax = plt.subplots()

visualize_sampleholder(sampleholder, ax, is_plot_contour=False, is_plot_hull=True)
plt.show()
print(sampleholder)

if False:
    cv2.imwrite("../images/fake_holder_with_samples_contours.jpg", image_to_visualize)
