""" 
The class of a contour of a sample
"""

from shapely.geometry import Polygon


class Contour:
    """
    The class of a contour of a sample
    """

    def __init__(self, contour, hull):
        self.contour = contour  # This is a cv2 contour
        self.hull = hull  # This is a cv2 contour
        self.polygon = Polygon(hull.squeeze().tolist())  # This is a shapely polygon
        self.sample = None  # This link to the sample object
        self.id = None  # this should be the same as the sample id
