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


def sample2polygon(sample: Sample):
    """given a sample, return the polygon"""
    return sample.contour_new.polygon


def is_polygons_overlap(polygon1: Polygon, polygon2: Polygon):
    return polygon1.intersects(polygon2)
