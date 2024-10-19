import numpy as np
from shapely.geometry import Polygon
from classes import Contour, FunctionalSampleHolder, Sample


def is_contours_overlap(contour1: Contour, contour2: Contour):
    return contour1.polygon.intersects(contour2.polygon)


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
