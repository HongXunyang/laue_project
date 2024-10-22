import numpy as np
from shapely.geometry import Polygon
from classes import Contour, FunctionalSampleHolder, Sample


def sampleholder2polygons(sampleholder: FunctionalSampleHolder):
    """
    given a sample holder, return a list of polygons
    """
    samples_list = sampleholder.samples_list
    return [sample.contour_new.polygon for sample in samples_list]


def sampleholder2vertices_list(sampleholder: FunctionalSampleHolder):
    """
    given a sample holder, return a list of vertices. A vertices is a (N, 2) numpy array, dtype=int32
    """
    samples_list = sampleholder.samples_list
    return [sample.contour_new.vertices for sample in samples_list]


def sample2polygon(sample: Sample):
    """given a sample, return the polygon"""
    return sample.contour_new.polygon


def is_two_polygons_overlap(polygon1: Polygon, polygon2: Polygon):
    return polygon1.intersects(polygon2)


def is_polygon_overlap_with_polygons(polygon: Polygon, polygons: list):
    for polygon_ in polygons:
        if is_two_polygons_overlap(polygon, polygon_):
            return True
    return False


def vertices_area(vertices: np.ndarray):
    """
    given a vertices, return the area of the polygon
    """
    return Polygon(vertices.tolist()).area


def update_sampleholder(sampleholder: FunctionalSampleHolder, new_vertices_list: list):
    """
    update the sampleholder with the new configuration
    """
    old_vertices_list = sampleholder2vertices_list(sampleholder)
    _update_sampleholder(sampleholder, old_vertices_list, new_vertices_list)
    return sampleholder


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


def _get_translation(old_vertices: np.ndarray, new_vertices: np.ndarray) -> np.ndarray:
    """get the translation vector by comparing the center of mass of the old and new vertices"""

    old_center = np.mean(old_vertices, axis=0)
    new_center = np.mean(new_vertices, axis=0)

    return new_center - old_center
