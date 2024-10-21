""" 
Package for processing image of samples and sample holder

Functions:
- remove_stripes: remove the underlying stripes from the image
- unify_background: unify the background color of the image
- contours2approximated_contours: approximate the contours using the Ramer-Douglas-Peucker algorithm
- contours2hulls: convert the contours to convex hulls
- image2contours: process the image and return the contours, approximated contours, and hulls
- generate_contour_object: generate a Contour object from the contour and hull
- generate_contour_objects: generate a list of Contour objects from the contours and hulls
- generate_sample_object: generate a Sample object from the contour and hull
- generate_sample_objects: generate a list of Sample objects from the contours and hulls
- generate_sampleholder_object: generate a SampleHolder object from the list of Sample objects

"""

import cv2
import numpy as np
import time
from classes import FunctionalSampleHolder, Sample, Contour
from classes import (
    _center_of_mass,
    _remove_background_contour,
    distance,
)


# --------------------------------
#  Main functions for processing
# --------------------------------
def remove_stripes(image, stripes_vectors, target_background_vector, isprint=False):
    """
    This function filters out the stripes from the image.

    Args:
    - image: cv2 image
    - (list) stripes_vectors: sampling of vectors that contains the BGR info of the stripes. [np.array([1,2,3]), np.array([255,1,136])] for examples
    - target_background_vector: np.array([1,2,3]) for example
    - isprint: if True, print the progress bar
    Returns: filtered_image
    ---------------

    # Mechanism
    - assuming v1, v2, v3 are the vectors of the stripes, v = (v1 + v2 + v3) / 3 is the center of mass.
    - R = max(distance(v, v1), distance(v, v2), distance(v, v3)) is the max distance from the center of mass to the stripes_vectors
    - For each pixel in the image, we calculate the distance to the v, denoted as r
    - If r < R, we replace the pixel with the background_vector
    """
    filtered_image = image.copy()
    mean_stripe_vector = _center_of_mass(stripes_vectors)
    R = max([distance(mean_stripe_vector, v) for v in stripes_vectors])

    diff_image = image - mean_stripe_vector
    diff_image_norm = np.linalg.norm(diff_image, axis=2)
    mask = diff_image_norm < 2 * R
    filtered_image[mask] = target_background_vector

    return filtered_image


def unify_background(
    image, background_vectors, target_background_vector, isprint=False
):
    """
    This function unifies the color of the background of the image

    Args:
    - image: cv2 image
    - (list of np.array) background_vectors: sampling of vectors that contains the BGR info of the background. [np.array([1,2,3]), np.array([255,1,136])] for examples
    - (np.array) target_background_vector: the target color of the background
    - isprint: if True, print the progress bar

    Returns: filtered_image
    ---------------

    # Mechanism
    - assuming v1, v2, v3 are the vectors sample of the background, v = (v1 + v2 + v3) / 3 is the center of mass.
    - R = max(distance(v, v1), distance(v, v2), distance(v, v3)) is the max distance from the center of mass to the background_vectors
    - For each pixel in the image, we calculate the distance to the center of mass v, denoted as r
    - If r < R, we replace the pixel with the target_background_vector
    """

    filtered_image = image.copy()
    mean_background_vector = _center_of_mass(background_vectors)
    R = max([distance(mean_background_vector, v) for v in background_vectors])
    diff_image = image - mean_background_vector
    diff_image_norm = np.linalg.norm(diff_image, axis=2)
    mask = diff_image_norm < 2 * R
    filtered_image[mask] = target_background_vector

    return filtered_image


def contours2approximated_contours(
    contours, epsilon=2.5, lowercut=100, area_lowercut=2000
):
    """
    This function approximates the contours using the Ramer-Douglas-Peucker algorithm.

    Args:
    - contours: list of contours
    - epsilon: the approximation accuracy
    - lowercut: the lowercut of the perimeter. If the perimeter of the contour is less than lowercut, we drop this contour. unit in pixel
    - area_lowercut: the lowercut of the area. If the area of the contour is less than area_lowercut, we drop this contour. unit in pixel^2

    Returns: the list of approximated contours, each contour's perimeter is larger than the lowercut
    ---------------
    """
    approximated_contours = []
    for i, contour in enumerate(contours):
        temp_contour = cv2.approxPolyDP(contour, epsilon, True)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        # if the perimeter is too small, or the approximated contour is a simple line, we drop it
        if (
            (perimeter > lowercut)
            and (len(temp_contour) > 2)
            and (area > area_lowercut)
        ):
            approximated_contours.append(temp_contour)
    return approximated_contours


def contours2hulls(contours):
    """
    This function approximates the contours using the convex hull algorithm.

    Args:
    - contours: list of contours

    Returns:
    - hulls: list of convex hulls
    - hulls_indeces: list of indices of the convex hull points
    """
    hulls = []  # To store convex hulls
    hulls_indeces = []  # To store indices of convex hull points

    for contour in contours:
        # Get hull points
        hull = cv2.convexHull(contour, returnPoints=True)
        hulls.append(hull)

        # Get hull point indices
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        hulls_indeces.append(hull_idx)

    return hulls, hulls_indeces


def image2contours(
    image,
    is_preprocess=True,
    stripes_vectors=None,
    background_vectors=None,
    epsilon=2.5,
    lowercut=100,
    area_lowercut=2000,
    gaussian_window=(5, 5),
    isprint=True,
    is_gaussian_filter=True,
):
    """
    This function process the image and return the original contours, approximated contours, and hulls

    Keyword arguments:
    - image: cv2 image
    - is_preprocess: if True, remove the stripes and unify the background
    - stripes_vectors: sampling of vectors that contains the BGR info of the stripes. [np.array([1,2,3]), np.array([255,1,136])] for examples
    - background_vectors: sampling of vectors that contains the BGR info of the background. [np.array([1,2,3]), np.array([255,1,136])] for examples
    - epsilon: the approximation accuracy
    - lowercut: the lowercut of the perimeter. If the perimeter of the contour is less than lowercut, we drop this contour. unit in pixel
    - area_lowercut: the lowercut of the area. If the area of the contour is less than area_lowercut, we drop this contour. unit in pixel^2
    - gaussian_window: the window size of the Gaussian filter
    - isprint: if True, print the progress bar
    - is_gaussian_filter: if True, apply Gaussian filter to the image
    """
    target_background_vector = np.mean(background_vectors, axis=0)
    target_background_vector = target_background_vector.astype(np.uint8)
    if is_preprocess:
        # raise error if stripes_vectors or background_vectors is None
        if stripes_vectors is None or background_vectors is None:
            raise ValueError("stripes_vectors or background_vectors is not provided")

        image_stripes_free = remove_stripes(
            image, stripes_vectors, target_background_vector, isprint
        )
        image_unfied_background = unify_background(
            image_stripes_free, background_vectors, target_background_vector, isprint
        )
        image_preprocessed = image_unfied_background
    else:
        image_preprocessed = image
    # convert to grayscale
    image_gray = cv2.cvtColor(image_preprocessed, cv2.COLOR_BGR2GRAY)
    # apply Gaussian filter
    if is_gaussian_filter:
        image_gray = cv2.GaussianBlur(image_gray, gaussian_window, 0)
    # binarize the image
    _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

    # get contours
    contours, hierarchy = cv2.findContours(
        image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # approximate contours
    approximated_contours = contours2approximated_contours(
        contours,
        epsilon=epsilon,
        lowercut=lowercut,
        area_lowercut=area_lowercut,
    )
    # get hulls
    hulls, _ = contours2hulls(approximated_contours)
    return (
        contours,
        approximated_contours,
        hulls,
    )


# -------------------------------------
#  Functions that produce class objects
# -------------------------------------
def generate_contour_object(contour, hull) -> Contour:
    """
    This function generates a Contour object from the contour, approximated contour, and hull.

    Args:
    - contour: the contour.
    - hull: the convex hull.

    Returns: Contour object
    ---------------
    """
    return Contour(contour, hull)


def generate_contour_objects(
    contours, hulls, is_remove_background_contour=True
) -> list:
    """
    This function generates a list of Contour objects from the contours, approximated contours, and hulls.

    Args:
    - contours: list of contours.
    - hulls: list of hulls.
    - is_remove_background_contour: if True, remove the background contour.
    Returns: list of Contour objects
    ---------------
    """
    if is_remove_background_contour:
        contours, hulls = _remove_background_contour(contours, hulls)
    contour_objects = []
    for i, (contour, approximated_contour, hull) in enumerate(zip(contours, hulls)):
        contour_objects.append(generate_contour_object(contour, hull))
    return contour_objects


def generate_sample_object(id, contour, hull, grid_index=None) -> Sample:
    """
    This function generates a Sample object from the contour, approximated contour, and hull.

    Args:
    - id: the id of the contour.
    - contour: the contour.
    - hull: the convex hull.

    Returns: Sample object
    ---------------
    """
    sample = Sample(id)
    sample.grid_index = grid_index

    # Generate the original contour objects
    contour_object_original = generate_contour_object(contour, hull)
    contour_object_original.sample = sample
    contour_object_original.id = id
    sample.contour_original = contour_object_original

    # Generate the new contour objects, which is the same as the original contour at the beginning
    contour_new, hull_new = contour.copy(), hull.copy()
    contour_object_new = generate_contour_object(contour_new, hull_new)
    contour_object_new.sample = sample
    contour_object_new.id = id
    sample.contour_new = contour_object_new

    # The position of the sample is the centroid of the hull
    sample.position_original = contour_object_original.center
    return sample


def generate_sample_objects(contours, hulls, is_remove_background_contour=True) -> list:
    """
    This function generates a list of Sample objects from the contours, approximated contours, and hulls.

    Args:
    - contours: list of contours.
    - hulls: list of hulls.
    - is_remove_background_contour: if True, remove the background contour.
    Returns: list of Sample objects. Usually, the 0th element is the background...
    ---------------
    """
    if is_remove_background_contour:
        contours, hulls = _remove_background_contour(contours, hulls)
    sample_objects = []

    for i, (contour, hull) in enumerate(zip(contours, hulls)):
        sample_objects.append(generate_sample_object(i, contour, hull))
    return sample_objects


def generate_sampleholder_object(samples) -> FunctionalSampleHolder:
    """
    This function generates a SampleHolder object from the list of Sample objects.

    Args:
    - samples: list of Sample objects.

    Returns: SampleHolder object
    ---------------
    """
    sampleholder = FunctionalSampleHolder()
    for sample in samples:
        sampleholder.add_sample(sample)
    return sampleholder
