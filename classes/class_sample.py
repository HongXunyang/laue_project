"""
This is a module that defines the class of sample
"""

import numpy as np
import cv2
from .helper_functions import _rotate, _contour2centroid

# To-do list:
# 1.  origin of the coordinate system should be defined;
# 2.  the origin of the sample itself should be defined


class Sample:
    def __init__(self, id: int, name="sample"):
        self.id = id
        self.name = name
        self.grid_index = None  # This is the grid index of the sample on the sample holder. (1,2) for example, python numeraction convention. None if grid not applicable
        self.sampleholder = None  # This link to the sample holder object
        self.contour_original = None  # This is the original contour of the sample (before re-orientation), found by CV
        self.contour_new = None  # new contour after re-orientation
        self.position_original: np.ndarray = (
            None  # (np.ndarray) Absolute position of the sample, currently defined as the centroid of the hull,
        )
        self.position_new: np.ndarray = (
            None  # (np.ndarray) New position after close packing.
        )

        # important properties
        self.phi_offset = None  # phi_offset of the sample, in degree, counter-clockwise
        self.is_reoriented = (
            False  # when the sample performes reorientation, this is set to True
        )
        self.is_relocated = (
            False  # when the sample performs relocation, this is set to True
        )

    def __str__(self):
        return f"Sample {self.id}, position: {self.position_original}"

    def reorient(self):
        """
        reorient sample according to the phi_offset
        """

        # contour_original, phi_offset must be assigned
        if self.contour_original is None:
            raise ValueError(f"Contour of sample {self.id} is not assigned")
        if self.phi_offset is None:
            raise ValueError(f"Phi offset of sample {self.id} is not assigned")

        # re-orient the sample
        center = self.position_original  # rotate the sample around the center
        contour_original = self.contour_original.contour
        hull_original = self.contour_original.hull
        for i in range(len(contour_original)):
            x_contour_original, y_contour_original = contour_original[i][0]
            x_contour_new, y_contour_new = _rotate(
                center, (x_contour_original, y_contour_original), self.phi_offset
            )
            self.contour_new.contour[i][0][0] = x_contour_new
            self.contour_new.contour[i][0][1] = y_contour_new

        for i in range(len(hull_original)):
            x_hull_original, y_hull_original = hull_original[i][0]
            x_hull_new, y_hull_new = _rotate(
                center, (x_hull_original, y_hull_original), self.phi_offset
            )
            self.contour_new.hull[i][0][0] = x_hull_new
            self.contour_new.hull[i][0][1] = y_hull_new

        # update the status of the sample (re-oriented)
        self.is_reoriented = True

    def relocate(self, is_print=False):
        """
        relocate the sample once the reorientation is done
        """

        # if the sample is not reoriented, raise an error
        if (not self.is_reoriented) and is_print:
            print(f"sample {self.id} is not re-oriented yet")
        # if the position_new is not assigned, raise an error
        if self.position_new is None:
            raise ValueError(
                f"The new position of sample {self.id} is not assigned, plz assign it first"
            )

        # relocate the sample
        # - move the contour and the hull to the new position
        translation = self.position_new - self.position_original
        self.contour_new.relocate(translation)
        self.is_relocated = True
