""" 
This package is for visualization
"""

import numpy as np
import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open("config/config.json", "r") as json_file:
    config = json.load(json_file)
with open("config/stylesheet.json", "r") as json_file:
    stylesheet = json.load(json_file)


def visualize_sampleholder(
    sampleholder, ax=None, is_plot_contour=False, is_plot_hull=True
):
    """
    This method visualizes the sample holder
    """
    for i, sample in enumerate(sampleholder.samples_list):
        contour_original = sample.contour_original.contour
        contour_new = sample.contour_new.contour
        hull_original = sample.contour_original.hull
        hull_new = sample.contour_new.hull

        # plot the original contour and hull using dashed line
        x_contour_original = [point[0][0] for point in contour_original]
        y_contour_original = [point[0][1] for point in contour_original]
        x_hull_original = [point[0][0] for point in hull_original]
        y_hull_original = [point[0][1] for point in hull_original]

        # plot the new contour and hull using solid line
        x_contour_new = [point[0][0] for point in contour_new]
        y_contour_new = [point[0][1] for point in contour_new]
        x_hull_new = [point[0][0] for point in hull_new]
        y_hull_new = [point[0][1] for point in hull_new]

        if ax is None:
            fig, ax = plt.subplots()

        if is_plot_contour:
            ax.plot(
                np.append(x_contour_original, x_contour_original[0]),
                np.append(y_contour_original, y_contour_original[0]),
                linestyle="--",
                color=np.array(stylesheet["contours_kwargs"]["color"])[::-1] / 255,
            )
        if is_plot_hull:
            ax.plot(
                np.append(x_hull_original, x_hull_original[0]),
                np.append(y_hull_original, y_hull_original[0]),
                linestyle="--",
                color=np.array(stylesheet["hulls_kwargs"]["color"])[::-1] / 255,
            )
        if is_plot_contour:
            ax.plot(
                np.append(x_contour_new, x_contour_new[0]),
                np.append(y_contour_new, y_contour_new[0]),
                linestyle="-",
                color=np.array(stylesheet["contours_kwargs"]["color"])[::-1] / 255,
            )
        if is_plot_hull:
            ax.plot(
                np.append(x_hull_new, x_hull_new[0]),
                np.append(y_hull_new, y_hull_new[0]),
                linestyle="-",
                color=np.array(stylesheet["hulls_kwargs"]["color"])[::-1] / 255,
            )
        ax.invert_yaxis()
        # add the id of the sample at the position of the sample
        if sample.is_relocated:
            text_position = sample.position_new

            # draw a link between the original position and the new position
            ax.arrow(
                sample.position_original[0],
                sample.position_original[1],
                sample.position_new[0] - sample.position_original[0],
                sample.position_new[1] - sample.position_original[1],
                head_width=10,
                head_length=10,
                fc="gray",
                ec="gray",
            )
            ax.scatter(
                sample.position_original[0],
                sample.position_original[1],
                marker="o",
                facecolors="gray",
                edgecolors="gray",
                s=7,
            )
        else:
            text_position = sample.position_original
        ax.text(
            text_position[0],
            text_position[1],
            sample.id,
            fontsize=12,
            color="black",
        )

    return ax
