""" 
The class of a contour of a sample
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate
from .helper_functions import _rotate


class Contour:
    """
    The class of a contour of a sample.
    """

    def __init__(self, contour, hull):
        self.contour = contour  # This is a cv2 contour. (N, 1, 2) numpy array, dtype = int32 initially, and change to float32 after relocation
        self.hull = hull  # This is a cv2 contour, (N, 1, 2) numpy array, dtype = int32 initially, and change to float32 after relocation
        self.polygon = Polygon(
            hull.squeeze().tolist()
        )  # This is a shapely polygon based on the convex hull
        self.vertices = (
            hull.squeeze()
        )  # the collection of vertices. (N, 2) numpy array, dtype = float32 from the very beginning
        self.vertices = self.vertices.astype(np.float32)
        self.area = self.polygon.area
        self.center = np.mean(self.vertices, axis=0)
        self.sample = None  # This link to the sample object
        self.id = None  # this should be the same as the sample id

    def reorient(self, phi_offset: float):
        self.contour = self.contour.astype(np.float32)
        self.hull = self.hull.astype(np.float32)

        # re-orient the sample
        center = self.center  # rotate the sample around the center
        contour_original = self.contour.copy()
        hull_original = self.hull.copy()
        for i in range(len(contour_original)):
            x_contour_original, y_contour_original = contour_original[i][0]
            x_contour_new, y_contour_new = _rotate(
                center, (x_contour_original, y_contour_original), phi_offset
            )
            self.contour[i][0][0] = x_contour_new
            self.contour[i][0][1] = y_contour_new

        for i in range(len(hull_original)):
            x_hull_original, y_hull_original = hull_original[i][0]
            x_hull_new, y_hull_new = _rotate(
                center, (x_hull_original, y_hull_original), phi_offset
            )
            self.hull[i][0][0] = x_hull_new
            self.hull[i][0][1] = y_hull_new

        self._update_vertices()

    def _update_vertices(self):
        # update the polygon and the vertices according to the current contour and hull
        self.polygon = Polygon(self.hull.squeeze().tolist())
        self.vertices = self.hull.squeeze()
        self.vertices = self.vertices.astype(np.float32)

    def relocate(self, translation: np.ndarray):
        # change the contour, hull, and vertices
        # convert all the int32 to float32 for better accuracy
        translation = translation.astype(np.float32)
        self.contour = self.contour.astype(np.float32)
        self.hull = self.hull.astype(np.float32)
        self.contour = self.contour + translation
        self.hull = self.hull + translation
        self.vertices = self.vertices + translation
        self.center = self.center + translation
        self.polygon = Polygon(self.hull.squeeze().tolist())
