""" 
This package is for optizing the confirguation of polygons.

TO-DO
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
    vertices_area,
)
from utils import visualize_vertices_list


def batch_optimization(
    sampleholder: FunctionalSampleHolder,
    number_system: int,
    is_plot=True,
    max_configurations: int = 9,
    number_of_iteration: int = 3000,
    step_size: int = 5,
    fluctuation: float = 0.1,
    temperature: float = 1000,
    is_rearrange_vertices=True,
    is_print=True,
    is_gravity=True,
    is_update_sampleholder=False,
):
    """
    Args:
    - number_system: number of system to run in parallel

    Kwargs:
    - is_plot: if True, plot the optimized configuration
    - max_configurations: the maximum number of configurations to plot
    - kwargs: the kwargs for optimization()
    """
    optimized_configuration_list = [None] * number_system
    area_list = np.zeros(number_system)
    for batch_index in range(number_system):
        optimized_configuration, area = optimization(
            sampleholder,
            number_of_iteration,
            step_size,
            fluctuation,
            temperature,
            is_rearrange_vertices,
            is_print,
            is_gravity,
            is_update_sampleholder,
        )
        optimized_configuration_list[batch_index] = optimized_configuration
        area_list[batch_index] = area

    if is_plot:
        if number_system == 1:
            fig, ax = plt.subplots()
            visualize_vertices_list(optimized_configuration_list[0], ax=ax)
        elif number_system > max_configurations:
            fig, axs = plt.subplots(3, 3, figsize=(20, 14))
            # sorted based on the area
            sorted_indices = np.argsort(area_list)
            for i, index in enumerate(sorted_indices[:max_configurations]):
                ax = axs[i // 3, i % 3]
                visualize_vertices_list(optimized_configuration_list[index], ax=ax)
                # on the left top corner, put the area of the configuration
                ax.text(
                    0.05,
                    0.95,
                    f"area:{area_list[index]:.3g}, index={index}",
                    transform=ax.transAxes,
                )
                ax.set(xticks=[], yticks=[])

        else:
            rows = int(np.ceil(np.sqrt(number_system)))
            columns = int(np.ceil(number_system / rows))
            fig, axs = plt.subplots(columns, rows, figsize=(20, 14))
            sorted_indices = np.argsort(area_list)
            for i, index in enumerate(sorted_indices):
                ax = axs[i % columns, i // columns]
                visualize_vertices_list(optimized_configuration_list[index], ax=ax)
                ax.text(
                    0.05,
                    0.95,
                    f"area:{area_list[index]:.3g}, index={index}",
                    transform=ax.transAxes,
                )
                ax.set(xticks=[], yticks=[])
    # ax setting: remove space between axes
    plt.subplots_adjust(wspace=0, hspace=0)


def optimization(
    sampleholder: FunctionalSampleHolder,
    number_of_iteration: int = 3000,
    step_size: int = 5,
    fluctuation: float = 0.1,
    temperature: float = 1000,
    is_rearrange_vertices=True,
    is_print=True,
    is_gravity=True,
    is_update_sampleholder=False,
):
    """
    Args:
    - sampleholder: sample holder

    Kwargs:
    - step_size: in pixel. How much a sample can move each step at max.
    - fluctuation: currently doing nothing
    - temperature: controling the posibilities of accepting inferior configuration
    - is_rearrange_vertices: if true, the initial positions of the samples will be rearranged for a better optimization.
    - is_print: if True, print out everything for debugging purpose
    - is_gravity: if True, the movement vector will be affected by the gravity of the samples
    - is_update_sampleholder: if True, the sampleholder will be modified/updated after the optimization

    Returns:
    - rearranged_vertices_list: the optimized configuration of the samples
    - area: the area of the convex hull of the optimized configuration
    """
    # preset annealing parameters
    initial_temperature = temperature
    current_temperature = temperature
    final_temperature = temperature * 0.01
    temperature_decay = (initial_temperature - final_temperature) / number_of_iteration
    step_size_decay = 0.8 * step_size / number_of_iteration

    # read polygons and convert them to list of vertices: list of (Nx2) np array, dtype= int32
    vertices_list = sampleholder2vertices_list(sampleholder)
    # rearrange the vertices_list for a better optimization (if is_rearrange_vertices is True)
    if is_rearrange_vertices:
        rearranged_vertices_list = _rearrange_vertices_list(vertices_list)

    area = _calculate_area(rearranged_vertices_list)  # initial area
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

    for iteration in range(number_of_iteration):
        # randomly select a polygon
        index = np.random.randint(0, number_polygons)
        vertices = rearranged_vertices_list[index]  # the vertices of the select polygon

        # create a movement vector
        # check if we accept the new configuration
        # (1) if there's overlap
        # (2) new configuration better than the previous one? with temperature effect
        # try to move the polygon, stored it in the temporary vertices
        movement_vector, direction_gravity = _create_movement_vector(
            rearranged_vertices_list,
            index,
            step_size,
            is_gravity,
            direction_gravity=None,
        )
        temp_vertices = vertices + movement_vector
        is_movement_allowed = _check_movement(temp_vertices, index, temp_vertices_list)
        attempt_counter = 0
        # even if not allowed, try 3 times
        while (not is_movement_allowed) and (attempt_counter < 3):
            movement_vector, gravity_direction = _create_movement_vector(
                rearranged_vertices_list,
                index,
                step_size // 2,
                is_gravity,
                direction_gravity=direction_gravity,
            )
            temp_vertices = vertices + movement_vector
            is_movement_allowed = _check_movement(
                temp_vertices, index, temp_vertices_list
            )
            attempt_counter += 1

        if is_movement_allowed:
            # if no overlap, update temp_vertices_list
            temp_vertices_list[index] = temp_vertices
            # check if the new configuration is better than the previous one
            temp_area, is_accept = _check_configuration(
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
        step_size -= step_size_decay  # linearly decrease the step_size

    if is_update_sampleholder:
        # at the end of the optimization, update the sample position by doing relocate()
        _update_sampleholder(sampleholder, vertices_list, rearranged_vertices_list)

    print(f"area = {area}")
    return rearranged_vertices_list, area


def _update_sampleholder(
    sampleholder: FunctionalSampleHolder, old_vertices_list, new_vertices_list: list
):
    """
    update the sampleholder with the new configuration
    """
    for i, vertices in enumerate(new_vertices_list):
        # determine the translation offset between the original and new vertices
        translation = _get_translation(old_vertices_list[i], new_vertices_list[i])
        sample = sampleholder.samples_list[i]
        # update the sample.position_new before applying sample.relocate()
        sample.position_new = sample.position_original + translation
        sample.relocate()


def _create_movement_vector(
    vertices_list: list,
    index: int,
    step_size: int,
    is_gravity=True,
    direction_gravity=None,
):
    """
    selection a direction and step size based on the configuration of polygons and also the temperature (randomness)

    Args:
    - vertices_list: list of (Nx2) np array, dtype= float32. This is the current configuration
    - index: the index of the sample you wanna move
    - step_size: how much the sample can move in both direction maximum. the actual movement will be a random number lying in the range (-step_size, +stepsize)
    - is_gravity: if True, the movement vector will be affected by the gravity of the samples
    """
    # the array of areas of the polygons in the vertices_list
    if is_gravity and direction_gravity is None:
        areas = np.array([vertices_area(vertices) for vertices in vertices_list])
        centers = np.array([np.mean(vertices, axis=0) for vertices in vertices_list])
        area_this, center_this = areas[index], centers[index]
        centers = centers - center_this
        distances_squared = np.sum(centers**2, axis=1)
        areas[index], distances_squared[index] = 0, 1
        gravities = centers / distances_squared[:, np.newaxis] * areas[:, np.newaxis]
        # select the direction based on the gravity
        direction_gravity = np.sum(gravities, axis=0)
        direction_gravity = direction_gravity / np.linalg.norm(
            direction_gravity
        )  # normalize the direction
    elif is_gravity and direction_gravity is not None:
        direction_gravity = direction_gravity
    else:
        direction_gravity = np.array([0, 0])
    # create a final movement vector
    movement_vector = direction_gravity * step_size + np.random.randint(
        -step_size, step_size, 2
    )

    return movement_vector, direction_gravity


def _check_movement(temp_vertices, index: int, vertices_list: list):
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


def _check_configuration(temp_vertices_list, area, temperature):
    """
    a function check if the configuration is better than the previous one

    Mechanism:
    - calculate the new area of the smallest circle-shape container that contains the temp_vertices_list
    - if the new area is smaller than the previous one, accept the new configuration
    - if the new area is larger than the previous one, accept the new configuration with a probability of exp(-(new_area - area)/temperature)
    """
    new_area = _calculate_area(temp_vertices_list)
    if new_area < area:
        is_accept = True
    else:
        probability = np.exp(-(new_area - area) / temperature)
        if np.random.rand() < probability:
            is_accept = True
        else:
            is_accept = False
    return new_area, is_accept


def _calculate_area(vertices_list):
    """
    calculate the area of the convex hull of the given vertices list
    """
    # Extract all points
    points = np.array([point for vertices in vertices_list for point in vertices])
    points = points.astype(np.int32)
    return cv2.contourArea(cv2.convexHull(points))


def _get_translation(old_vertices: np.ndarray, new_vertices: np.ndarray) -> np.ndarray:
    """get the translation vector by comparing the center of mass of the old and new vertices"""

    old_center = np.mean(old_vertices, axis=0)
    new_center = np.mean(new_vertices, axis=0)

    return new_center - old_center


def _rearrange_vertices_list(
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
