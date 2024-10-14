""" 
This module defines the class of the sample holder 
"""

import numpy as np
import matplotlib.pyplot as plt
from .sample_class import Sample


class SampleHolder:
    def __init__(self, grid_size=(10, 10)):
        self.name = "Sample Holder"
        self.grid_size = (
            grid_size  # This is the index range of the grid on the sample holder
        )
        self.size = (
            None  # this is the actual dimension/size of the sample holder, in mm
        )
        self.grid = np.zeros(
            self.grid_size
        )  # This is the true table of the sample holder: 0 -> no sample; 1 -> with sample
        self.samples = [
            [None for _ in range(grid_size[1])] for _ in range(grid_size[0])
        ]  # This is the list storing the sample objects
        self._index2id = [
            [None for _ in range(grid_size[1])] for _ in range(grid_size[0])
        ]  # given the index, return the id of the sample
        self._id2index = {}  # given the id, return the index of the sample

    def __str__(self):
        return f"Sample Holder: {self.name}, grid size: {self.grid_size}"

    # Methods definition
    def add_sample(self, sample: Sample):
        """
        This method adds a sample to the sample holder

        Args:
        --------------
        sample: Sample
            The sample object to be added to the sample holder
        """
        index = sample.index
        self.samples[index[0]][index[1]] = sample  # assign sample to the grid
        # update the grid true table; 0 -> no sample; 1 -> with sample
        self.grid[index[0]][index[1]] = 1
        self._index2id[index[0]][index[1]] = sample.id
        self._id2index[str(sample.id)] = index
        sample.sample_holder = self

        return sample

    def remove_sample(self, id: int):
        """
        remove sample by id
        1. set the index in the grid to zero
        2. remove the sample from the sample list (set to None)
        3. remove the id from the id2index dictionary
        4. remove the index from the index2id list
        """
        index = self.id2index(id)
        self.grid[index[0]][index[1]] = 0
        self.samples[index[0]][index[1]] = None
        self._id2index.pop(str(id))
        self._index2id[index[0]][index[1]] = None

    # helper methods
    def id2index(self, id):
        """
        This method returns the index of the sample given the id

        Args:
        --------------
        id: int
            The id of the sample

        Returns:
        --------------
        index: tuple
            The index of the sample on the sample holder
        """
        return self._id2index[str(id)]

    def index2id(self, index):
        """
        This method returns the id of the sample given the index

        Args:
        --------------
        index: tuple
            The index of the sample on the sample holder

        Returns:
        --------------
        id: int
            The id of the sample
        """
        return self._index2id[index[0]][index[1]]

    def number_samples(self):
        """
        This method returns the number of samples on the sample holder

        Returns:
        --------------
        num_samples: int
            The number of samples on the sample holder
        """
        return int(np.sum(np.array(self.grid)))

    def is_sample(self, index: tuple) -> bool:
        """
        given the index, return if there is a sample on the sample holder
        """
        return self.grid[index[0]][index[1]] == 1

    # Visualization methods
    def visualize(self, ax=None):
        """
        This method visualizes the sample holder
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
                if self.is_sample((i, j)):  # if sample exists
                    ax.plot(j, i, "ro", markersize=10)

        ax.invert_yaxis()  # invert y axis
        ax.xaxis.tick_top()  # move x ticks to the top

        # adding id next to the sample
        offset = 0.12
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.is_sample((i, j)):  # if sample exists
                    ax.text(j + offset, i + offset, self._index2id[i][j], fontsize=12)

        # plot the phi offset by adding a small line on the sample dot
        line_length = 0.5
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.is_sample((i, j)):  # if sample exists
                    sample = self.samples[i][j]
                    if sample.phi_offset is not None:
                        phi_offset = sample.phi_offset
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
