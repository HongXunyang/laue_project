import pytest
import numpy as np
import cv2
from close_packing import optimization, batch_optimization
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import save_sampleholder


def test_optimization():
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
    image = cv2.imread("data/test_image.jpg")
    rows, columns, _ = image.shape

    # compress image
    image = cv2.resize(image, (rows // 4, columns // 4), interpolation=cv2.INTER_AREA)
    rows, columns, _ = image.shape
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
    samples_list = generate_sample_objects(approximated_contours, hulls)
    sampleholder = generate_sampleholder_object(samples_list)
    # ----------- end of contour finding ----------- #
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
    filename = "test_sampleholder.json"
    sampleholder_dict = save_sampleholder(sampleholder, folder_path, filename)
