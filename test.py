from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time, cv2
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import animate_config_evolution, save_sampleholder, visualize_sampleholder
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from config.config import (
    batch_optimization_kwargs,
    tests_config,
    image2contours_kwargs,
)
from close_packing import batch_optimization, optimization

STEP_CONTROL = dict(
    test=False, contour_finding=True, close_packing=True, convert_to_cad=True
)
stripes_vectors = [[124, 128, 147], [105, 115, 133], [104, 115, 142]]


# ----------- image pre-processing ----------- #
if STEP_CONTROL["contour_finding"] or STEP_CONTROL["test"]:
    start_time = time.time()
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


plt.show()
