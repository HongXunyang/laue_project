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
from config.config import (
    physical_size,
    batch_optimization_kwargs,
    config,
    tests_config,
    image2contours_kwargs,
)


start_time = time.time()
# ----------- image pre-processing ----------- #
stripes_vectors, background_vectors, target_background_vector = (
    tests_config["stripes_vectors"],
    tests_config["background_vectors"],
    tests_config["target_background_vector"],
)
# Load image
image = cv2.imread(tests_config["test_image_path"])
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
)


end_time = time.time()
print(f"image processed time: {end_time - start_time} seconds\n")

# create samples objects and sample holder object
samples_list = generate_sample_objects(approximated_contours, hulls)
sampleholder = generate_sampleholder_object(samples_list)

# ----------- end of contour finding ----------- #
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------------- optimization --------------- #

start_time = time.time()
optimized_configuration_list, area_list, sorted_indices, _ = batch_optimization(
    sampleholder,
    **batch_optimization_kwargs,
)
end_time = time.time()
print(f"optimization time: {end_time - start_time} seconds\n")

# ----------- end of optimization ----------- #
# # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------- convert Samples to CAD --------- #

vertices_list = optimized_configuration_list[sorted_indices[0]]
# Save the configuration of the samples to a CAD (STL) file
# by default, the output CAD files will be saved in the `temporary_output/` folder.
vertices_list_to_cad(
    vertices_list,
)
# ------------------- end  --------------------------- #
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------- convert Sample holder to CAD ----------- #

# update sampleholder object
sampleholder.shape = "circle"
sampleholder.update()

# Save the engraved sampleholder to a CAD (STL) file
# by default, the output CAD files will be saved in the `temporary_output/` folder.
sampleholder_to_cad(
    sampleholder,
)
# ----------- end  ----------- #
plt.show()
