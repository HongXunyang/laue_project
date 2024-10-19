""" 
The class of a contour of a sample
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate


class Contour:
    """
    The class of a contour of a sample.
    """

    def __init__(self, contour, hull):
        self.contour = (
            contour  # This is a cv2 contour. (N, 1, 2) numpy array, dtype = int32
        )
        self.hull = hull  # This is a cv2 contour, (N, 1, 2) numpy array, dtype = int32
        self.polygon = Polygon(
            hull.squeeze().tolist()
        )  # This is a shapely polygon based on the convex hull
        self.vertices = (
            hull.squeeze()
        )  # the collection of vertices. (N, 2) numpy array, dtype = int32
        self.area = self.polygon.area
        self.sample = None  # This link to the sample object
        self.id = None  # this should be the same as the sample id

    def relocate(self, translation: np.ndarray):
        # change the contour, hull, and vertices
        translation = translation.astype(np.int32)
        self.contour = self.contour + translation
        self.hull = self.hull + translation
        self.vertices = self.vertices + translation
