"""
This is a module that defines the class of sample
"""

import numpy as np
import cv2

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
        self.posiition_original = None  # Absolute position of the sample
        self.position_new = None  # New position after close packing.

        # important properties
        self.phi_offset = None  # phi_offset of the sample, in degree, counter-clockwise
        self.is_reoriented = False
        self.is_relocated = False

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

    def relocate(self):
        """
        relocate the sample once the reorientation is done
        """

        # if the sample is not reoriented, raise an error
        if not self.is_reoriented:
            raise ValueError(
                f"Sample {self.id} is not reoriented, plz reorient it first"
            )
        # if the position_new is not assigned, raise an error
        if self.position_new is None:
            raise ValueError(
                f"The new position of sample {self.id} is not assigned, plz assign it first"
            )

        # relocate the sample
        # - move the contour and the hull to the new position
        x_offset = self.position_new[0] - _contour2centroid(self.contour_new.contour)[0]
        y_offset = self.position_new[1] - _contour2centroid(self.contour_new.contour)[1]
        for i in range(len(self.contour_new.contour)):
            self.contour_new.contour[i][0][0] += x_offset
            self.contour_new.contour[i][0][1] += y_offset
        for i in range(len(self.contour_new.hull)):
            self.contour_new.hull[i][0][0] += x_offset
            self.contour_new.hull[i][0][1] += y_offset

        self.is_relocated = True


def _rotate(center, point, phi_offset):
    """Rotate a point around the center, compenate the phi_offset

    Args:
    - center: center of the rotation
    - point: point to rotate
    - phi_offset: the angle to rotate, in degree, counter-clockwise
    """
    x, y = point  # -1, 0
    cx, cy = center  # 0, 0
    # target 0, 1
    # phi_offset = 90
    phi_to_rotate = -phi_offset * np.pi / 180
    x_new = (x - cx) * np.cos(phi_to_rotate) - (y - cy) * np.sin(phi_to_rotate) + cx
    y_new = (x - cx) * np.sin(phi_to_rotate) + (y - cy) * np.cos(phi_to_rotate) + cy
    return x_new, y_new


def _contour2centroid(contour):
    M = cv2.moments(contour)
    # Ensure the area is not zero before calculating centroid
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        print(f"Contour has zero area, returning None.")
        return None
