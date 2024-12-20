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
import cv2, os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import time
from classes import FunctionalSampleHolder
from utils import (
    sampleholder2vertices_list,
    vertices_area,
    update_sampleholder,
    visualize_vertices_list,
    visualize_sampleholder,
    visualize_area_evolution,
    save_sampleholder,
)
from config import physical_size, config


def batch_optimization(
    sampleholder: FunctionalSampleHolder,
    number_system: int,
    is_print=True,
    number_of_iterations: int = 10000,
    step_size: int = 20,
    temperature: float = 300,
    contour_buffer_multiplier: float = 1.0,
    optimize_shape="convex_hull",
    is_rearrange_vertices=True,
    is_gravity=True,
    gravity_multiplier: float = 0.5,
    gravity_off_at: int = 3000,
    is_contour_buffer=True,
    is_save_results=True,
    is_record_area_history=True,
    ax_area=None,
    progress_callback=None,
):
    """
    Args:
    - number_system: number of system to run in parallel

    Kwargs:
    - is_plot: if True, plot the optimized configuration
    - is_print: if True, print stuff for debugging
    - is_save_results: if True, save the results in the temporary_output folder
    - progress_callback: ...?
    - kwargs: the kwargs for optimization()

    Returns:
    - optimized_configuration_list: a list of optimized vertices_list
    - area_list: a list of the area of the convex hull of the optimized configuration
    - sorted_indices: the indices of the optimized_configuration_list sorted based on the area
    """
    # initialization
    max_configurations = 9  # the maximum number of configurations to plot

    ## optimized_configuration_list = [None] * number_system
    best_configuration = None  # the best configuration ever
    best_area = np.inf  # the best area ever
    area_list = np.zeros(number_system)
    start_time = time.time()
    iteration_times = []
    if ax_area is None:
        fig_area, ax_area = plt.subplots()
        ax_area.set_title("Area Evolution")
        ax_area.set_xlabel("Iteration")
        ax_area.set_ylabel("area")
    ax_ratio = ax_area.twinx()

    area_evolution_list = [None] * number_system
    sampleholder.update()

    # ---------------- start the optimization ---------------- #
    for batch_index in range(number_system):
        iteration_start_time = time.time()
        if is_print:
            print(f"NO.{batch_index+1} out of {number_system} started")

        optimized_configuration, optimized_area, optimization_history = optimization(
            sampleholder,
            number_of_iterations,
            step_size=step_size,
            temperature=temperature,
            contour_buffer_multiplier=contour_buffer_multiplier,
            optimize_shape=optimize_shape,
            is_rearrange_vertices=is_rearrange_vertices,
            is_gravity=is_gravity,
            gravity_multiplier=gravity_multiplier,
            gravity_off_at=gravity_off_at,
            is_update_sampleholder=False,
            is_contour_buffer=is_contour_buffer,
            is_plot_evolution=False,
            is_record_area_history=is_record_area_history,
            is_record_configuration_history=False,
        )
        area_evolution_list[batch_index] = optimization_history["area_evolution"]
        if optimized_area < best_area:
            best_area = optimized_area
            best_configuration = optimized_configuration

        ## optimized_configuration_list[batch_index] = optimized_configuration
        area_list[batch_index] = optimized_area
        sorted_indices = np.argsort(area_list)

        # estimate time
        iteration_time = time.time() - iteration_start_time
        iteration_times.append(iteration_time)

        # Estimate total time based on average iteration time
        elapsed_time = time.time() - start_time
        average_iteration_time = sum(iteration_times) / len(iteration_times)
        estimated_total_time = average_iteration_time * number_system
        remaining_time = estimated_total_time - elapsed_time

        if progress_callback is not None:
            progress = ((batch_index + 1) * 1.0) / number_system * 100
            progress_callback(progress, estimated_total_time, remaining_time)
    # ---------------- end the optimization ---------------- #

    # update the sample holder anyway
    new_vertices_list = best_configuration
    update_sampleholder(sampleholder, new_vertices_list)

    # ------- plot the optimized configuration ------- #
    fig_config, ax_config = plt.subplots()
    visualize_sampleholder(sampleholder, ax=ax_config)
    fig_config.tight_layout()
    # ------------------------------------------------ #

    # ----------------- Plot the area evolution ----------------- #
    ax_area, ax_ratio = visualize_area_evolution(
        sampleholder=sampleholder,
        area_evolution_list=area_evolution_list,
        ax_area=ax_area,
        ax_ratio=ax_ratio,
    )
    fig_area.tight_layout()
    # ----------------------------------------------------------- #

    # ----------------- Save the results ----------------- #
    # check folder
    if is_save_results:
        if not os.path.exists(config["temporary_output_folder"]):
            os.makedirs(config["temporary_output_folder"])

        # save the optimized configuration plot
        optimized_configuration_path = os.path.join(
            config["temporary_output_folder"], "optimized_configuration.jpg"
        )
        fig_config.savefig(optimized_configuration_path, dpi=200)

        # save the area evolution plot
        area_evolution_path = os.path.join(
            config["temporary_output_folder"], "area_evolution.jpg"
        )
        fig_area.savefig(area_evolution_path, dpi=200)

        # save the sampleholder. change the name of the output file within the folder if the results are desirable
        save_sampleholder(sampleholder)

    return best_configuration, area_list, sorted_indices, area_evolution_list


def optimization(
    sampleholder: FunctionalSampleHolder,
    number_of_iterations: int = 3000,
    step_size: int = 20,
    temperature: float = 1000,
    contour_buffer_multiplier: float = 1.0,
    optimize_shape="min_circle",
    is_rearrange_vertices=True,
    is_gravity=True,
    gravity_multiplier: float = 0.5,
    gravity_off_at: int = 3000,
    is_update_sampleholder=False,
    is_contour_buffer=True,
    is_plot_evolution=False,
    is_record_configuration_history=False,
    is_record_area_history=True,
    ax_area=None,
    ax_ratio=None,
):
    """
    Args:
    - sampleholder: sample holder

    Kwargs:
    - step_size: in pixel. How much a sample can move each step at max.
    - temperature: controling the posibilities of accepting inferior configuration
    - contour_buffer_multiplier: The contour buffer is a buffer around the convex hull of each sample. The buffer is used to avoid edge touching of samples. 1.01 means the convex hull of the samples will be 1% percent larger than its actual size. The larger the buffer, the larger the space between the samples.
    - optimize_shape: the shape of the area to optimize. Choose from "convex_hull" or "min_circle"
    - is_rearrange_vertices: if true, the initial positions of the samples will be rearranged for a better optimization.
    - is_gravity: if True, the movement vector will be affected by the gravity of the samples. This will increase the running time by roughly 50%.
    - gravity_multiplier: controling the strength of the gravity. 1 means the movement vector is always along the gravity direction; 0.5 means the movement vector is somewhat along the gravity direction; 1.5 means the movement vector is more along the gravity direction.
    - gravity_off_at: the iteration number when the gravity effect is turned off. The gravity effect is turned off by setting gravity_multiplier to 0.
    - is_update_sampleholder: if True, the sampleholder will be modified/updated after the optimization
    - is_contour_buffer: if True, the contour of the samples will be inflated by a small amount to create buffer area betwee nsamples, avoiding edge touching
    - is_plot_evolution: if True, plot out the area evolution during the optimization process
    - is_record_history: record the history of the vertices_list during the optimization process
    - ax_area: the axis to plot the area evolution
    - ax_ratio: the axis to plot the ratio of the sample area to the area of the sampleholder

    Returns:
    - best_vertices_list: the best optimized (ever) configuration of the samples
    - best_area: the area of sampleholder of the best configuration
    - optimization_history: a dictionary containing the area_evolution and vertices_list_evolution
    """
    # initialization
    sampleholder.contour_buffer_multiplier = contour_buffer_multiplier
    sampleholder.is_contour_buffer = is_contour_buffer
    # preset annealing parameters
    initial_temperature = temperature
    current_temperature = temperature
    temperature_at_gravity_off = temperature * 0.2
    # temperature decays exponentially from the initial_temperature to the temperature_at_gravity_off at gravity_off_at step.
    temperature_decay_rate = np.exp(
        np.log(temperature_at_gravity_off / initial_temperature) / gravity_off_at
    )  # temperature decays exponentially
    step_size_decay = 0.5 * step_size / number_of_iterations

    # read polygons and convert them to list of vertices: list of (Nx2) np array, dtype= int32
    vertices_list = sampleholder2vertices_list(sampleholder)
    if is_contour_buffer:
        vertices_list = _inflate_vertices_list(
            vertices_list=vertices_list, multiplier=contour_buffer_multiplier
        )

    # here stores the best configuration ever
    best_vertices_list = vertices_list.copy()

    # rearrange the vertices_list for a better optimization (if is_rearrange_vertices is True)
    if is_rearrange_vertices:
        rearranged_vertices_list = _rearrange_vertices_list(vertices_list)
    else:
        rearranged_vertices_list = vertices_list.copy()

    # the area of each sample
    sample_areas_list = np.array(
        [vertices_area(vertices) for vertices in vertices_list]
    )
    sampleholder.update()
    samples_area = sampleholder.samples_area

    # the initial area of the sampleholder
    area = _calculate_area(
        rearranged_vertices_list, shape=optimize_shape
    )  # initial area
    best_area = area  # the best area ever

    # ---------------- History recording variables ---------------- #
    if is_plot_evolution or is_record_area_history:
        area_evolution = np.zeros(number_of_iterations)
        area_evolution[0] = area
    else:
        area_evolution = None

    if is_record_configuration_history:
        vertices_list_evolution = [None] * number_of_iterations
        vertices_list_evolution[0] = rearranged_vertices_list.copy()
    else:
        vertices_list_evolution = None
    # -----------------------------------------------------------#

    scale_hull = np.sqrt(area)  # the scale of the convex hull
    ideal_temperature = (
        scale_hull * step_size
    )  # the order of magnitude of the initial temperature

    # create a temporary vertices_list to store the temporary new vetices
    temp_vertices_list = rearranged_vertices_list.copy()
    number_polygons = len(vertices_list)

    # -------- Start of the optimization -------- #
    for iteration in range(number_of_iterations):
        # randomly select a polygon
        index = np.random.randint(0, number_polygons)
        vertices = rearranged_vertices_list[index]  # the vertices of the select polygon
        if iteration > gravity_off_at:
            is_gravity = False
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
            areas=sample_areas_list,
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
                areas=sample_areas_list,
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

                # update the best configuration ever
                if best_area > area:
                    best_area = area
                    best_vertices_list = rearranged_vertices_list.copy()
            else:
                # if not accepted, revert the temp_vertices_list
                temp_vertices_list[index] = vertices

        current_temperature = (
            current_temperature * temperature_decay_rate
        )  # exponentially decrease the temperature
        step_size -= step_size_decay  # linearly decrease the step_size
        if is_plot_evolution or is_record_area_history:
            area_evolution[iteration] = area
        if is_record_configuration_history:
            vertices_list_evolution[iteration] = rearranged_vertices_list.copy()
            if is_contour_buffer:
                vertices_list_evolution[iteration] = _inflate_vertices_list(
                    vertices_list=vertices_list_evolution[iteration],
                    multiplier=1 / contour_buffer_multiplier,
                )
    # -------- End of the optimization -------- #

    if is_contour_buffer:
        # remove the buffer area by deflating the vertices again
        best_vertices_list = _inflate_vertices_list(
            vertices_list=best_vertices_list, multiplier=1 / contour_buffer_multiplier
        )

    if is_update_sampleholder:
        update_sampleholder(sampleholder, best_vertices_list)

    # ----------------- Plot the area evolution ----------------- #
    if is_plot_evolution:
        if (ax_area is None) or (ax_ratio is None):
            fig, ax_area = plt.subplots()
            ax_ratio = ax_area.twinx()
        visualize_area_evolution(sampleholder, area_evolution, ax_area, ax_ratio)
    # ----------------------------------------------------------- #

    optimization_history = dict(
        area_evolution=area_evolution,
        vertices_list_evolution=vertices_list_evolution,
    )
    return best_vertices_list, best_area, optimization_history


def _create_movement_vector(
    vertices_list: list,
    index: int,
    step_size: int,
    is_gravity=True,
    gravity_multiplier: float = 0.7,
    direction_gravity=None,
    areas=None,
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
    - areas: the areas of the polygons in the vertices_list. If None, the areas will be calculated based on the vertices_list
    """
    # the array of areas of the polygons in the vertices_list
    if is_gravity and direction_gravity is None:
        if areas is None:
            areas = np.array([vertices_area(vertices) for vertices in vertices_list])
        centers = np.array([np.mean(vertices, axis=0) for vertices in vertices_list])
        center_this = centers[index]
        centers = centers - center_this
        distances_squared = np.sum(centers**2, axis=1)
        distances_squared[index] = 1000.0
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


def _check_configuration(temp_vertices_list, area, temperature, shape="min_circle"):
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


def _calculate_area(vertices_list, shape="min_circle"):
    """
    calculate the area of the convex hull of the given vertices list

    kwargs:
    - shape: the shape of the area to calculate. Choose from "convex_hull" or "min_circle"

    Note:
    - if shape == "min_ciecle", the are is not the real area, but is replaced by the radius of the minimum enclosing circle
    """
    # Extract all points
    points = np.array([point for vertices in vertices_list for point in vertices])
    points = points.astype(np.float32)
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
        new_vertices_list[index] = new_vertices_list[index].astype(np.float32)
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
