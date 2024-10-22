""" 
This module defines the class of the sample holder. There are two types of sample holder
1. Grid sample holder: an sample holder with grid on 
2. Functional sample holder: an advanced sample holder that can perform re-orientation of sample and close packing of samples. This is a bridge between the plain sample holder and the engraved sample holder
3. Engraved sample holder: an sample holder with sample contour engraved on it

# setup of the class definition
1. Basic sample holder class that share the same properties between the two types
3. Functional sample holder class that inherits from the basic sample holder class. 
2. grid sample holder class that inherits from the functional sample holder class
4. Engraved sample holder class that inherits from the functional sample holder class
"""

import numpy as np
import json, cv2
import matplotlib.pyplot as plt
from .class_sample import Sample
from .helper_functions import _sampleholder2vertices_list


# Load the data from the JSON file
with open("config/config.json", "r") as json_file:
    config = json.load(json_file)
with open("config/stylesheet.json", "r") as json_file:
    stylesheet = json.load(json_file)


# Basic sample holder class
class SampleHolder:
    """
    the class of the sample holder. Samples are labeled by the list_index, 0, 1 for example. list_index is assigned to a sample once it is added to the sample holder. Sample objects are stored in the samples_list. Each sample should have a unique id (int).
    """

    def __init__(self):
        self.name = "Sample Holder"
        self.shape: str = None  # choose from "ciecle", "rectangle"
        self.size: list = (
            None  # This is the actual dimension/size [width, height] of the sample holder, in mm, [100, 100] for example
        )
        self.radius: float = None  # the radius of the sample holder if it is a circle
        self.thickness: float = None  # the thickness of the sample holder
        self.center: np.ndarray = None  # the center position of the sample holder
        self.convex_hull = None  # the cv2 convex hull of the sample holder
        self.min_circle = None  # the minimum enclosing circle of the convex hull
        self.samples_list = []  # This is the list storing the sample objects
        self._id2sample = {}  # given the id, return the sample object
        self._id2list_index = (
            {}
        )  # given the id, return the index of the sample in the list

    def __str__(self):
        # return the name of the sample holder
        # return the number of samples on the sample holder
        return f"{self.name} with {self.number_samples()} samples"

    def print_samples(self):
        for sample in self.samples_list:
            print(sample)

    # Core methods
    def add_sample(self, sample: Sample):
        """
        This method adds a sample to the basic sample holder

        Args:
        --------------
        sample: Sample
            The sample object to be added to the sample holder
        """
        self.samples_list.append(sample)
        sample.sampleholder = self
        self._id2sample[str(sample.id)] = sample
        self._id2list_index[str(sample.id)] = len(self.samples_list) - 1

    # ---------------------------------------------
    # helper methods
    # ---------------------------------------------
    def update_convex_hull(self):
        """
        update the convex hull based on the current sample configuration
        """
        vertices_list = _sampleholder2vertices_list(self)
        points = np.array([point for vertices in vertices_list for point in vertices])
        points = points.astype(np.float32)
        self.convex_hull = cv2.convexHull(points)

    def update_min_circle(self):
        """
        update the minimum enclosing circle based on the current convex hull
        """
        if self.convex_hull is None:
            self.update_convex_hull()
        center, radius = cv2.minEnclosingCircle(self.convex_hull)
        self.center = np.array(center)
        self.radius = radius
        return center, radius

    def vertices_list(self):
        return _sampleholder2vertices_list(self)

    def id2sample(self, id: int):
        """
        This method returns the sample object given the id

        Args:
        --------------
        id: int
            The id of the sample

        Returns:
        --------------
        sample: Sample
            The sample object
        """
        return self._id2sample[str(id)]

    def id2list_index(self, id: int):
        """
        This method returns the list index of the sample in the list given the id

        Args:
        --------------
        id: int
            The id of the sample

        Returns:
        --------------
        index: int
            The index of the sample in the list
        """
        return self._id2list_index[str(id)]

    def list_index2id(self, index: int):
        """
        This method returns the id of the sample given the list index

        Args:
        --------------
        index: int
            The index of the sample in the list

        Returns:
        --------------
        id: int
            The id of the sample, or None if the index is out of range
        """
        if index >= len(self.samples_list):
            return None
        return self.samples_list[index].id

    def number_samples(self):
        """
        This method returns the number of samples on the sample holder

        Returns:
        --------------
        num_samples: int
            The number of samples on the sample holder
        """
        return len(self.samples_list)


class FunctionalSampleHolder(SampleHolder):
    """
    This class defines the functional sample holder that can perform re-orientation of sample and close packing of samples.
    """

    def __init__(self):
        super().__init__()
        self.name = "Functional Sample Holder"

    # ---------------------------------------------
    # Core methods
    # - assign_phi_offset: this method assigns the phi offset to the sample and update the sample.phi_offset
    # - reorient_sample: this method re-orients the sample according to the sample.phi_offset, and update their sample.contour_new
    # - relocate_samples: this method pack the samples on the same sample holder, leave the contour unchanged but update the sample.position_new
    # ---------------------------------------------

    def assign_phi_offset(self, sample: Sample, phi_offset: float):
        """
        assigns the phi offset to the sample and update the sample.phi_offset
        """
        sample.phi_offset = phi_offset

    def assign_phi_offset_by_index(self, index, phi_offset: float, search_type="id"):
        """
        assigns the phi offset to the sample given the index.

        keyword arguments:
        ----------------
        search_type: str
            the type of index to search for the sample. choose from "id" or "list_index"
        """
        if search_type == "id":
            sample = self.id2sample(index)
            self.assign_phi_offset(sample, phi_offset)
        elif search_type == "list_index":
            sample = self.samples_list[index]
            self.assign_phi_offset(sample, phi_offset)

    def reorient_sample(self, sample: Sample):
        """
        re-orient the sample according to the phi_offset and update the sample.contour_new
        """
        # if phi offset is not assigned, raise an error
        if sample.phi_offset is None:
            raise ValueError(f"The phi offset of sample {sample.id} is not assigned")
        sample.reorient()
        pass

    def reorient_sample_by_index(self, index, search_type="id"):
        """
        re-orient the sample given the index
        """
        if search_type == "id":
            sample = self.id2sample(index)
            self.reorient_sample(sample)
        elif search_type == "list_index":
            sample = self.samples_list[index]
            self.reorient_sample(sample)

    def reorient(self):
        """
        re-orient all samples on the sample holder
        """
        for sample in self.samples_list:
            self.reorient_sample(sample)

    def relocate_sample(self, sample: Sample, position):
        """
        relocate the sample to the new position
        """
        pass

    def relocate_sample_by_index(self, index, position, search_type="id"):
        """
        relocate the sample given the index
        """
        if search_type == "id":
            sample = self.id2sample(index)
            self.relocate_sample(sample, position)
        elif search_type == "list_index":
            sample = self.samples_list[index]
            self.relocate_sample(sample, position)

    def rescale(self):
        """
        rescale the size of everything
        - rescale the sample holder size, radius, thickness
        - rescale the sample position, contour size
        """
        pass


# (Currently not used!!!)
# (Under development!!!)
# Grid sample holder class with grid on
class GridSampleHolder(FunctionalSampleHolder):
    """
    This class defines the grid sample holder with grid on.
    Samples are labeled by the grid_index, (0,2) for example, in addtion to the list_index defined in the parent class. grid_index is pre-defined in the sample object when constructured. In this class, sample objects are stored both in the samples_list and in the samples_in_grid.

    Enable sample update (re-orientation, close packing) by grid index.
    """

    def __init__(self, grid_size=(10, 10)):
        super().__init__()
        self.name = "Grid Sample Holder"
        self.grid_size = (
            grid_size  # This is the index range of the grid on the sample holder
        )
        self._is_sample_in_grid = np.zeros(
            self.grid_size, dtype=np.int8
        )  # This is the true table of the sample holder: 0 -> no sample; 1 -> with sample
        self.samples_in_grid = [
            [None for _ in range(grid_size[1])] for _ in range(grid_size[0])
        ]  # This is the list storing the sample objects, indexed by the grid index, (0,2) for example
        self._id2grid_index = (
            {}
        )  # given the id, return the grid index of the sample in the grid
        self._grid_index2id = [
            [None for _ in range(grid_size[1])] for _ in range(grid_size[0])
        ]  # given the grid index (in tuple), return the id of the sample

    # ---------------------------------------------
    # core methods
    # - add_sample
    # - reorient_sample
    # - relocate_sample
    # ---------------------------------------------
    def add_sample(self, sample: Sample):
        """
        This method adds a sample to the grid sample holder

        Args:
        --------------
        sample: Sample
            The sample object to be added to the sample holder
        """
        super().add_sample(sample)
        grid_index = sample.grid_index
        self._is_sample_in_grid[grid_index[0]][grid_index[1]] = 1
        self.samples_in_grid[grid_index[0]][grid_index[1]] = sample
        # make sure the id does not exist in the dictionary before adding
        if str(sample.id) in self._id2grid_index:
            raise ValueError("The id already exists in the dictionary")
        else:
            self._id2grid_index[str(sample.id)] = grid_index  # from id to grid index
        self._grid_index2id[grid_index[0]][
            grid_index[1]
        ] = sample.id  # from grid index to id

    def reorient_sample_by_index(self, index, search_type="grid_index"):
        super().reorient_sample_by_index(index, search_type)
        if search_type == "grid_index":
            sample = self.samples_in_grid[index[0]][index[1]]
            self.reorient_sample(sample)

    def relocate_sample_by_index(self, index, position, search_type="grid_index"):
        super().relocate_sample_by_index(index, position, search_type)
        if search_type == "grid_index":
            sample = self.samples_in_grid[index[0]][index[1]]
            self.relocate_sample(sample, position)

    # ---------------------------------------------
    # helper methods
    # ---------------------------------------------
    def is_sample_in_grid(self, index: tuple) -> bool:
        """
        given the index, return if there is a sample on the sample holder
        """
        return self._is_sample_in_grid[index[0]][index[1]] == 1

    def id2grid_index(self, id: int):
        """
        This method returns the grid index of the sample given the id

        Args:
        --------------
        id: int
            The id of the sample

        Returns:
        --------------
        grid index: tuple
            The grid index of the sample on the sample holder
        """
        return self._id2grid_index[str(id)]

    def grid_index2id(self, grid_index: tuple):
        """
        This method returns the id of the sample given the index

        Args:
        --------------
        grid_index: tuple
            The grid index of the sample on the sample holder. (1,2) for example

        Returns:
        --------------
        id: int
            The id of the sample
        """
        return self._grid_index2id[grid_index[0]][grid_index[1]]

    # --------------------------------------------
    # Visualization methods
    # --------------------------------------------
    def visualize_abstract(self, ax=None):
        """
        This method visualizes the sample holder in an abstract way (grid + dots)
        """

        # plotting the grid and add a circle for each sample
        # 1. plot the grid
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect("equal")

        grid_size = self.grid_size
        for i in range(grid_size[0] + 1):
            ax.axhline(i, color="gray", lw=1, alpha=0.5)
        for i in range(grid_size[1] + 1):
            ax.axvline(i, color="gray", lw=1, alpha=0.5)

        ax.set(
            xlim=(-1, grid_size[1]),
            ylim=(-1, grid_size[0]),
            xticks=np.arange(0, grid_size[1], 1),
            yticks=np.arange(0, grid_size[0], 1),
        )

        # plot the samples as dots if exists
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.is_sample_in_grid((i, j)):  # if sample exists
                    ax.plot(j, i, "ro", markersize=10)

        ax.invert_yaxis()  # invert y axis
        ax.xaxis.tick_top()  # move x ticks to the top

        # adding id next to the sample
        offset = 0.12
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.is_sample_in_grid((i, j)):  # if sample exists
                    ax.text(
                        j + offset, i + offset, self._grid_index2id[i][j], fontsize=12
                    )

        # plot the phi offset by adding a small line on the sample dot
        line_length = 0.5
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.is_sample_in_grid((i, j)):  # if sample exists
                    sample = self.samples_in_grid[i][j]
                    if sample.phi_offset is not None:
                        phi_offset = sample.phi_offset
                        if sample.is_oriented:
                            phi_offset = 0
                        x = j - line_length / 2 * np.cos(np.deg2rad(phi_offset))
                        x1 = j + line_length / 2 * np.cos(np.deg2rad(phi_offset))
                        y = i - line_length / 2 * np.sin(np.deg2rad(phi_offset))
                        y1 = i + line_length / 2 * np.sin(np.deg2rad(phi_offset))
                        ax.plot(
                            [x, x1],
                            [y, y1],
                            "gray",
                            zorder=-1,
                            linewidth=2.5,
                            alpha=0.7,
                        )
        return ax
