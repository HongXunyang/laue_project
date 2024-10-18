""" 
This package is for optizing the confirguation of polygons.
"""

import numpy as np
import cv2
from shapely.affinity import translate
from shapely.geometry import Polygon
from classes import FunctionalSampleHolder
from .helper_functions import (
    is_contours_overlap,
    sampleholder2polygons,
    sample2polygon,
    is_polygon_overlap_with_polygons,
)
from .visualization import visualize_vertices_list


def optimization(
    sampleholder: FunctionalSampleHolder = None,
    number_of_iteration: int = 1000,
    step_size: float = 0.1,
    fluctuation: float = 0.1,
    temperature: float = 0.1,
    is_plot: bool = True,
):

    area = 1000000  # area of the container, we wanna minimize this.
    # read polygons
    if sampleholder is None:
        # if no sampleholder is provided, we create a random vertices_list
        vertices_list = np.array(
            [
                [
                    [point for point in np.random.randint(0, 100, 2) + i * 30]
                    for _ in range(3)
                ]
                for i in range(2)
            ]
        )

        temp_vertices_list = vertices_list.copy()
        number_polygons = 2
    else:
        polygons = sampleholder2polygons(sampleholder)
        vertices_list = [list(polygon.exterior.coords) for polygon in polygons]
        temp_vertices_list = vertices_list.copy()
        number_polygons = len(polygons)

    if is_plot:
        visualize_vertices_list(vertices_list)

    for iteration in range(number_of_iteration):
        # randomly select a polygon
        index = np.random.randint(0, number_polygons)
        vertices = np.array(vertices_list[index])
        # create a movement vector
        movement_vector = _create_movement_vector(
            vertices_list, index, step_size, fluctuation
        )
        # try to move the polygon
        temp_vertices = vertices + movement_vector

        # check if we accept the new configuration
        # - check if there's overlap
        # - check if the new configuration is better than the previous one, with temperature effect of course
        if check_movement(temp_vertices, index, temp_vertices_list):
            temp_vertices_list[index] = temp_vertices.tolist()

            temp_area, is_accept = check_configuration(
                temp_vertices_list, area, temperature
            )
            if is_accept:
                area = temp_area
                vertices_list[index] = temp_vertices.tolist()
            else:
                temp_vertices_list[index] = vertices.tolist()

    # after the optimization, update the sampleholder
    # - calculate the x,y offset of each sample
    # - update the position_new of each sample
    # - run the relocation function of each sample (this function might need to be updated)

    # for now, return the new vertices_list
    if is_plot:
        visualize_vertices_list(vertices_list)
    return vertices_list


def _create_movement_vector(polygons, index: int, step_size: float, flucuation: float):
    """
    selection a direction and step size based on the configuration of polygons and also the temperature (randomness)
    """
    # at the moment it's just random
    return np.random.rand(2) * 5


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
    calculate the area of the smallest circle-shape container that contains the vertices_list
    """
    # Extract all points
    points = np.array([point for vertices in vertices_list for point in vertices])
    points = points.astype(np.int32)
    center, radius = cv2.minEnclosingCircle(points)
    return np.pi * radius**2
