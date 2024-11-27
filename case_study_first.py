from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time, cv2
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import visualize_sampleholder, rearrange_samples_indeces, visualize_contours
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from config.config import (
    batch_optimization_kwargs,
    tests_config,
    image2contours_kwargs,
)
from close_packing import batch_optimization, optimization


mm_per_pixel = 5.0 / (1450.0 - 1282.0)


# ----------- image pre-processing ----------- #

start_time = time.time()
stripes_vectors = np.array([[124, 128, 147], [105, 115, 133], [104, 115, 142]])
background_vectors = np.array([[175, 169, 162], [184, 178, 171], [189, 181, 174]])
target_background_vector = np.array([180, 175, 168])

# Load image
image = cv2.imread("../images/samples_from_jasmin.jpg")
rows, columns, channels = image.shape

# ----------- end of image pre-processing ----------- #
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ---------------- contour finding ------------------ #
# Find the contours of the samples in the image
# by default, the output images will be saved in the `temporary_output/` folder.
contours, approximated_contours, hulls, _ = image2contours(
    image,
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    **image2contours_kwargs,
    is_output_image=True,
)


end_time = time.time()
print(f"image processed time: {end_time - start_time} seconds\n")

# create samples objects and sample holder object
samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)
contours = [sample.contour_original.contour for sample in sampleholder.samples_list]
hulls = [sample.contour_original.hull for sample in sampleholder.samples_list]


visualize_contours(
    image,
    contours,
    hulls,
    is_plot=False,
    is_output_image=True,
)


# ----------- end of contour finding ----------- #
visualize_sampleholder(sampleholder, is_min_circle=False)


#