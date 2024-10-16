""" 
The class of a contour of a sample
"""


class Contour:
    """
    The class of a contour of a sample
    """

    def __init__(self, contour, hull):
        self.contour = contour
        self.hull = hull
        self.sample = None  # This link to the sample object
        self.id = None  # this should be the same as the sample id
