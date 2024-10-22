""" 
This package is for optizing the confirguation of polygons. 

Algorithm: Simulated Annealing
Object function: the area of the convex hull of the configuration. The smaller the bertter.

TO-DO:
----------
- make sample rotatable during the optimization process, based on the symmetry of the sample
- enable absolute buffer area setting between samples. Currently, the buffer area is relative to the size of the samples.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from classes import FunctionalSampleHolder
from utils import (
    sampleholder2vertices_list,
    vertices_area,
    update_sampleholder,
    visualize_vertices_list,
)


def batch_optimization(
    sampleholder: FunctionalSampleHolder,
    number_system: int,
    is_plot=True,
    is_print=True,
    number_of_iteration: int = 3000,
    step_size: int = 5,
    fluctuation: float = 0.1,
    temperature: float = 1000,
    contour_buffer_multiplier: float = 1.01,
    optimize_shape="convex_hull",
    is_rearrange_vertices=True,
    is_gravity=True,
    gravity_multiplier: float = 0.5,
    is_update_sampleholder=False,
    is_contour_buffer=True,
    is_plot_area=False,
    ax_area=None,
):
    """
    Args:
    - number_system: number of system to run in parallel

    Kwargs:
    - is_plot: if True, plot the optimized configuration
    - is_print: if True, print stuff for debugging
    - kwargs: the kwargs for optimization()

    Returns:
    - optimized_configuration_list: a list of optimized vertices_list
    - area_list: a list of the area of the convex hull of the optimized configuration
    - sorted_indices: the indices of the optimized_configuration_list sorted based on the area
    """
    # initialization
    max_configurations = 9  # the maximum number of configurations to plot
    optimized_configuration_list = [None] * number_system
    area_list = np.zeros(number_system)
    if is_plot_area:
        fig, ax_area = plt.subplots()
        ax_area.set_title("Area Evolution")
        ax_area.set_xlabel("Iteration")
        ax_area.set_ylabel("area")
    # start the optimization
    for batch_index in range(number_system):
        if is_print:
            print(f"NO.{batch_index+1} out of {number_system} started")
        optimized_configuration, area = optimization(
            sampleholder,
            number_of_iteration,
            step_size=step_size,
            fluctuation=fluctuation,
            temperature=temperature,
            contour_buffer_multiplier=contour_buffer_multiplier,
            optimize_shape=optimize_shape,
            is_rearrange_vertices=is_rearrange_vertices,
            is_gravity=is_gravity,
            gravity_multiplier=gravity_multiplier,
            is_update_sampleholder=False,
            is_contour_buffer=is_contour_buffer,
            is_plot_area=is_plot_area,
            ax_area=ax_area,
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

    # update the sample holder if is_update_sampleholder is True
    if is_update_sampleholder:
        new_vertices_list = optimized_configuration_list[sorted_indices[0]]
        update_sampleholder(sampleholder, new_vertices_list)

    return optimized_configuration_list, area_list, sorted_indices


def optimization(
    sampleholder: FunctionalSampleHolder,
    number_of_iteration: int = 3000,
    step_size: int = 5,
    fluctuation: float = 0.1,
    temperature: float = 1000,
    contour_buffer_multiplier: float = 1.01,
    optimize_shape="convex_hull",
    is_rearrange_vertices=True,
    is_gravity=True,
    gravity_multiplier: float = 0.5,
    is_update_sampleholder=False,
    is_contour_buffer=True,
    is_plot_area=False,
    ax_area=None,
):
    """
    Args:
    - sampleholder: sample holder

    Kwargs:
    - step_size: in pixel. How much a sample can move each step at max.
    - fluctuation: currently doing nothing
    - temperature: controling the posibilities of accepting inferior configuration
    - contour_buffer_multiplier: The contour buffer is a buffer around the convex hull of each sample. The buffer is used to avoid edge touching of samples. 1.01 means the convex hull of the samples will be 1% percent larger than its actual size. The larger the buffer, the larger the space between the samples.
    - optimize_shape: the shape of the area to optimize. Choose from "convex_hull" or "min_circle"
    - is_rearrange_vertices: if true, the initial positions of the samples will be rearranged for a better optimization.
    - is_gravity: if True, the movement vector will be affected by the gravity of the samples
    - gravity_multiplier: controling the strength of the gravity. 1 means the movement vector is always along the gravity direction; 0.5 means the movement vector is somewhat along the gravity direction; 1.5 means the movement vector is more along the gravity direction.
    - is_update_sampleholder: if True, the sampleholder will be modified/updated after the optimization
    - is_contour_buffer: if True, the contour of the samples will be inflated by a small amount to create buffer area betwee nsamples, avoiding edge touching
    - is_plot_area: if True, plot out the area evolution during the optimization process
    - ax_area: the axis to plot the area evolution

    Returns:
    - rearranged_vertices_list: the optimized configuration of the samples
    - area: the area of the convex hull of the optimized configuration
    """
    # initialization

    # preset annealing parameters
    initial_temperature = temperature
    current_temperature = temperature
    final_temperature = temperature * 0.01
    temperature_decay = (initial_temperature - final_temperature) / number_of_iteration
    step_size_decay = 0.8 * step_size / number_of_iteration

    # read polygons and convert them to list of vertices: list of (Nx2) np array, dtype= int32
    vertices_list = sampleholder2vertices_list(sampleholder)

    # inflate the vertices to leave buffer area between samples.
    if is_contour_buffer:
        vertices_list = _inflate_vertices_list(
            vertices_list=vertices_list, multiplier=contour_buffer_multiplier
        )
    # rearrange the vertices_list for a better optimization (if is_rearrange_vertices is True)
    if is_rearrange_vertices:
        rearranged_vertices_list = _rearrange_vertices_list(vertices_list)

    area = _calculate_area(
        rearranged_vertices_list, shape=optimize_shape
    )  # initial area
    if is_plot_area:
        area_evolution = np.zeros(number_of_iteration)
        area_evolution[0] = area
    scale_hull = np.sqrt(area)  # the scale of the convex hull
    ideal_temperature = (
        scale_hull * step_size
    )  # the order of magnitude of the initial temperature

    # create a temporary vertices_list to store the temporary new vetices
    temp_vertices_list = rearranged_vertices_list.copy()
    number_polygons = len(vertices_list)

    # -------- Start of the optimization -------- #
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
            gravity_multiplier=gravity_multiplier,
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
                gravity_multiplier=gravity_multiplier,
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
                temp_vertices_list, area, current_temperature, shape=optimize_shape
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
        if is_plot_area:
            area_evolution[iteration] = area
    # -------- End of the optimization -------- #

    if is_contour_buffer:
        # remove the buffer area by deflating the vertices again
        vertices_list = _inflate_vertices_list(
            vertices_list=vertices_list, multiplier=1 / contour_buffer_multiplier
        )
        rearranged_vertices_list = _inflate_vertices_list(
            vertices_list=rearranged_vertices_list,
            multiplier=1 / contour_buffer_multiplier,
        )

    if is_update_sampleholder:
        # at the end of the optimization, update the sample position by doing relocate()
        update_sampleholder(sampleholder, rearranged_vertices_list)

    if is_plot_area:
        ax_area.plot(np.array(range(number_of_iteration)), np.log(area_evolution))
    return rearranged_vertices_list, area


def _create_movement_vector(
    vertices_list: list,
    index: int,
    step_size: int,
    is_gravity=True,
    gravity_multiplier: float = 0.7,
    direction_gravity=None,
):
    """
    selection a direction and step size based on the configuration of polygons and also the temperature (randomness)

    Args:
    - vertices_list: list of (Nx2) np array, dtype= float32. This is the current configuration
    - index: the index of the sample you wanna move
    - step_size: how much the sample can move in both direction maximum. the actual movement will be a random number lying in the range (-step_size, +stepsize)
    - is_gravity: if True, the movement vector will be affected by the gravity of the samples
    - graivity_multiplier: the multiplier of the gravity strength. 1 means the created movement vector is always somewhat along the gravity direction (inner product > 0); 0.5 means weaker gravity effect (inner product could < 0); 1.5 means strong gravity effect (more along the gravit direction)
    -  direction_gravity: the direction of the gravity. If None, the direction will be calculated based on the configuration of the polygons.
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
    movement_vector = (
        direction_gravity * step_size
    ) * gravity_multiplier + np.random.randint(-step_size, step_size, 2)

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


def _check_configuration(temp_vertices_list, area, temperature, shape="convex_hull"):
    """
    a function check if the configuration is better than the previous one

    Mechanism:
    - calculate the new area of the smallest circle-shape container that contains the temp_vertices_list
    - if the new area is smaller than the previous one, accept the new configuration
    - if the new area is larger than the previous one, accept the new configuration with a probability of exp(-(new_area - area)/temperature)
    """
    new_area = _calculate_area(temp_vertices_list, shape=shape)
    if new_area < area:
        is_accept = True
    else:
        probability = np.exp(-(new_area - area) / temperature)
        if np.random.rand() < probability:
            is_accept = True
        else:
            is_accept = False
    return new_area, is_accept


def _calculate_area(vertices_list, shape="convex_hull"):
    """
    calculate the area of the convex hull of the given vertices list

    kwargs:
    - shape: the shape of the area to calculate. Choose from "convex_hull" or "min_circle"

    Note:
    - if shape == "min_ciecle", the are is not the real area, but is replaced by the radius of the minimum enclosing circle
    """
    # Extract all points
    points = np.array([point for vertices in vertices_list for point in vertices])
    points = points.astype(np.int32)
    convex_hull = cv2.convexHull(points)
    if shape == "convex_hull":
        return cv2.contourArea(convex_hull)
    elif shape == "min_circle":
        # calculate the minimum enclosing circle of the convex hull
        _, radius = cv2.minEnclosingCircle(convex_hull)
        return radius

    else:
        raise ValueError(f"please choose shape from 'convex_hull' or 'min_circle'.")


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


def _inflate_vertices(vertices: np.ndarray, multiplier: float = 1.01):
    """
    inflate the vertices by a small amount

    Mechanism:
    - calculate the center of the vertices
    - move the vertices away from the center by (1-multiplier)
    """

    center = np.mean(vertices, axis=0)
    return center + (vertices - center) * multiplier


def _inflate_vertices_list(vertices_list: list, multiplier: float = 1.01):
    """
    inflate each vertices in the vertices list by a small amount

    Args:
    - vertices_list: a list of (Nx2) np array, dtype=float32

    Kwarg:
    - multiplier: the multiplier of the inflation. 1.01 means the vertices will be inflated by 1%
    """
    return [_inflate_vertices(vertices, multiplier) for vertices in vertices_list]
