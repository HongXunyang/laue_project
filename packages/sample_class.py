"""
This is a module that defines the class of sample
"""

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
        self.phi_offset = None
        self.is_reoriented = False

    def __str__(self):
        return f"Sample {self.id}, position: {self.position_original}"
