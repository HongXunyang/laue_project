""" 
This package is for visualization
"""

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from .helper_functions import _remove_background_contour

# Load the data from the JSON file
with open("config/config.json", "r") as json_file:
    config = json.load(json_file)
with open("config/stylesheet.json", "r") as json_file:
    stylesheet = json.load(json_file)


def visualize_sampleholder(
    sampleholder, ax=None, is_plot_contour=False, is_plot_hull=True
):
    """
    This method visualizes the sample holder.
    - Plot the original contours with dashed lines
    - Plot the new contours after reorientation or relocation with solid lines
    - indicate the movement of the samples with arrows

    Keyword arguments:
    - ax: the axis to plot the sample holder.
    - is_plot_contour: if True, plot the contours.
    - is_plot_hull: if True, plot the hulls.

    Returns:
    - ax: the axis with the sample holder plotted.
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


def visualize_contours(
    image,
    contours,
    hulls,
    is_remove_background_contour=True,
    is_plot=True,
):
    """
    Overlay the contours and hulls on the image and label the contours.

    Args:
    - image: cv2 image (the image on which to draw the contours and hulls).
    - contours: list of contours.
    - hulls: list of hulls.

    Keyword arguments:
    - contours_kwargs: the kwargs for the contours (e.g., {"color": (0, 255, 0), "thickness": 2}).
    - hulls_kwargs: the kwargs for the hulls (e.g., {"color": (255, 0, 0), "thickness": 2}).
    - is_remove_background_contour: if True, remove the background contour.
    - is_plot: if True, plot the image.
    """
    if is_remove_background_contour:
        contours, hulls = _remove_background_contour(contours, hulls)

    image_copy = image.copy()

    # Draw contours on the image
    cv2.drawContours(image_copy, contours, -1, **stylesheet["contours_kwargs"])

    # Draw hulls on the image
    cv2.drawContours(image_copy, hulls, -1, **stylesheet["hulls_kwargs"])

    # Label the contours according to their index
    number_contours = len(contours)
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        # Ensure the area is not zero before calculating centroid
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # the color of the font depends on the i value: (255,255,255) while for i=0, and linearly decreases down to (150,150,150) gray
            font_color = (
                int(255 - 105 * i / number_contours),
                int(255 - 105 * i / number_contours),
                int(255 - 105 * i / number_contours),
            )
            # the font size depends on the i value: 1.8 while for i=0, and linearly decreases down to 1.0
            font_size = 1.8 - 0.8 * i / number_contours
            cv2.putText(
                image_copy,
                str(i),  # Label with contour index
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,  # Font scale
                font_color,
                3,  # Thickness
            )
        else:
            print(f"Contour {i} has zero area, skipping text label.")
    if is_plot:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result", 800, 500)
        cv2.imshow("result", image_copy)
    return image_copy
