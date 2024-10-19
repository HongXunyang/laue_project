""" 
This package is for optizing the confirguation of polygons.

TO-DO
- slowly decrease the temperature
- create a emsable with multiple system running at the same time 
- adjustable step_size, step_size also depends on the mass of the sample: the smaller the faster
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from classes import FunctionalSampleHolder
from .helper_functions import (
    sampleholder2vertices_list,
)
from utils import visualize_vertices_list


def optimization(
    sampleholder: FunctionalSampleHolder,
    number_of_iteration: int = 1000,
    step_size: int = 5,
    fluctuation: float = 0.1,
    temperature: float = 300,
    is_plot: bool = True,
    is_rearrange_vertices=True,
    is_print=True,
):
    """
    Args:
    - sampleholder: sample holder

    Kwargs:
    - step_size: in pixel. How much a sample can move each step at max.
    - fluctuation: currently doing nothing
    - temperature: controling the posibilities of accepting inferior configuration
    - is_plot: if True, plot out the initial and final configuration
    - is_rearrange_vertices: if true, the initial positions of the samples will be rearranged for a better optimization.
    - is_print: if True, print out everything for debugging purpose
    """
    # preset annealing parameters
    initial_temperature = temperature
    current_temperature = temperature
    final_temperature = temperature * 0.01
    # temperature decays linearly
    temperature_decay = (initial_temperature - final_temperature) / number_of_iteration

    # read polygons and convert them to list of vertices: list of (Nx2) np array, dtype= int32
    vertices_list = sampleholder2vertices_list(sampleholder)
    # rearrange the vertices_list for a better optimization (if is_rearrange_vertices is True)
    if is_rearrange_vertices:
        rearranged_vertices_list = rearrange_vertices_list(vertices_list)

    area = calculate_area(rearranged_vertices_list)  # initial area
    scale_hull = np.sqrt(area)  # the scale of the convex hull
    ideal_temperature = (
        scale_hull * step_size
    )  # the order of magnitude of the initial temperature
    if is_print:
        print(
            f"the ideal order of temperature is around {ideal_temperature/1.5:.1f}\nthe initial temperature is {initial_temperature}"
        )

    # create a temporary vertices_list to store the temporary new vetices
    temp_vertices_list = rearranged_vertices_list.copy()
    number_polygons = len(vertices_list)

    if is_plot:
        # initial configuraion
        visualize_vertices_list(vertices_list)

    for iteration in range(number_of_iteration):
        # randomly select a polygon
        index = np.random.randint(0, number_polygons)
        vertices = rearranged_vertices_list[index]  # the vertices of the select polygon
        # create a movement vector
        movement_vector = _create_movement_vector(
            rearranged_vertices_list, index, step_size
        )
        # try to move the polygon, stored it in the temporary vertices
        temp_vertices = vertices + movement_vector

        # check if we accept the new configuration
        # (1) if there's overlap
        # (2) new configuration better than the previous one? with temperature effect
        if check_movement(temp_vertices, index, temp_vertices_list):
            # if no overlap, update temp_vertices_list
            temp_vertices_list[index] = temp_vertices
            # check if the new configuration is better than the previous one
            temp_area, is_accept = check_configuration(
                temp_vertices_list, area, current_temperature
            )
            if is_accept:
                # new configuration is accepted, update the configuration
                area = temp_area
                rearranged_vertices_list[index] = temp_vertices
            else:
                # if not accepted, revert the temp_vertices_list
                temp_vertices_list[index] = vertices

        current_temperature -= temperature_decay  # linearly decrease the temperature

    if is_plot:
        visualize_vertices_list(rearranged_vertices_list)

    # at the end of the optimization, update the sample position by doing relocate()
    for i, vertices in enumerate(rearranged_vertices_list):
        # determine the translation offset between the original and new vertices
        translation = get_translation(vertices_list[i], rearranged_vertices_list[i])
        sample = sampleholder.samples_list[i]
        # update the sample.position_new before applying sample.relocate()
        sample.position_new = sample.position_original + translation
        sample.relocate(is_print=is_print)

    print(f"area = {area}")
    return rearranged_vertices_list


def _create_movement_vector(vertices_list: list, index: int, step_size: int):
    """
    selection a direction and step size based on the configuration of polygons and also the temperature (randomness)

    Args:
    - vertices_list: list of (Nx2) np array, dtype= int32. This is the current configuration
    - index: the index of the sample you wanna move
    - step_size: how much the sample can move in both direction maximum. the actual movement will be a random number lying in the range (-step_size, +stepsize)
    """

    return np.random.randint(-step_size, step_size, 2)


def check_movement(temp_vertices, index: int, vertices_list: list):
    """
    a function check if the movement is valid or not
    """
    # check if the new polygon overlap with other polygons
    # create polygons
    #
    temp_polygon = Polygon(temp_vertices)
    for i, vertices in enumerate(vertices_list):
        if i == index:
            continue
        polygon = Polygon(vertices)
        if temp_polygon.intersects(polygon):
            return False
    return True


def check_configuration(temp_vertices_list, area, temperature):
    """
    a function check if the configuration is better than the previous one

    Mechanism:
    - calculate the new area of the smallest circle-shape container that contains the temp_vertices_list
    - if the new area is smaller than the previous one, accept the new configuration
    - if the new area is larger than the previous one, accept the new configuration with a probability of exp(-(new_area - area)/temperature)
    """
    new_area = calculate_area(temp_vertices_list)
    if new_area < area:
        is_accept = True
    else:
        probability = np.exp(-(new_area - area) / temperature)
        if np.random.rand() < probability:
            is_accept = True
        else:
            is_accept = False
    return new_area, is_accept


def calculate_area(vertices_list):
    """
    calculate the area of the convex hull of the given vertices list
    """
    # Extract all points
    points = np.array([point for vertices in vertices_list for point in vertices])
    points = points.astype(np.int32)
    return cv2.contourArea(cv2.convexHull(points))


def get_translation(old_vertices: np.ndarray, new_vertices: np.ndarray) -> np.ndarray:
    """get the translation vector by comparing the center of mass of the old and new vertices"""

    old_center = np.mean(old_vertices, axis=0)
    new_center = np.mean(new_vertices, axis=0)

    return new_center - old_center


def rearrange_vertices_list(
    vertices_list: list, block_size_multiplier: int = 1.5
) -> list:
    """
    re-arrange the vertices list randomly for a better optimization.

    Kwargs:
    - block_size_multiplier: how many times the subblock is bigger than the scale of the sample.

    Mechanism:
    - Determin the largest scale (1d, in pixel) of the samples, denoted by `scale_sample`
    - setup a grid, each subblock/subregion in the grid is of the size of `2*scale_sample`
    - randomly assign sample to the center of those subregion
    - check overlap, if True, redo the process

    """

    is_overlap = True
    number_vertices = len(vertices_list)
    scale_sample = _determine_largest_scale(vertices_list)
    block_size = block_size_multiplier * scale_sample
    while is_overlap:
        # visualize_vertices_list(vertices_list, ax=ax)
        temp_vertices_list = _randomly_rearrange(vertices_list, block_size)
        # visualize_vertices_list(temp_vertices_list, ax=ax)
        is_overlap = _check_overlap(temp_vertices_list)

    return temp_vertices_list


def _determine_largest_scale(vertices_list: list) -> int:
    """determine the largest scale of a list of vertices

    Mechanism:
    - calcualte the area of the convex hull of each sample
    - pick out the largest area A
    - the scale would be the square root of A
    """
    for index, vertices in enumerate(vertices_list):
        if index == 0:
            scale = np.sqrt(cv2.contourArea(cv2.convexHull(vertices)))
            temp_scale = scale
        else:
            temp_scale = np.sqrt(cv2.contourArea(cv2.convexHull(vertices)))
            if temp_scale > scale:
                scale = temp_scale

    return int(scale)


def _check_overlap(vertices_list: list):
    return False


def _randomly_rearrange(vertices_list: list, block_size: int):
    number_vertices = len(vertices_list)
    center_list = [[np.mean(vertices, axis=0)] for vertices in vertices_list]

    rows = int(np.sqrt(number_vertices)) + 1
    numbers_to_select = np.arange(0, rows * rows, 1)
    order_list = np.random.choice(
        numbers_to_select, size=number_vertices, replace=False
    )

    new_center_list = [
        np.array([order // rows, order % rows]) * block_size for order in order_list
    ]
    new_vertices_list = vertices_list.copy()
    for index, vertices in enumerate(vertices_list):
        old_center = np.mean(vertices, axis=0)
        translation = new_center_list[index] - old_center
        new_vertices_list[index] = vertices_list[index] + translation

    return new_vertices_list
