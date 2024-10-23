import numpy as np
import json
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from classes import _remove_background_contour


# Load the data from the JSON file
with open("config/config.json", "r") as json_file:
    config = json.load(json_file)
with open("config/stylesheet.json", "r") as json_file:
    stylesheet = json.load(json_file)


def visualize_sampleholder(
    sampleholder,
    ax=None,
    is_only_new=True,
    is_plot_contour=False,
    is_plot_hull=True,
    is_fill_new_polygon=True,
    is_relocation_arrow=False,
    is_min_circle=True,
):
    """
    This method visualizes the sample holder.
    - Plot the original contours with dashed lines
    - Plot the new contours after reorientation or relocation with solid lines
    - indicate the movement of the samples with arrows

    Keyword arguments:
    - ax: the axis to plot the sample holder.
    - is_only_new: if True, only plot the new contours and hulls
    - is_plot_contour: if True, plot the contours of each samples
    - is_plot_hull: if True, plot the hulls of each samples
    - is_fill_new_polygon: if True, fill the new polygon with color
    - is_relocation_arrow: if True, an arrow pointing from the original position to the new position will be ploted for each sample
    - is_min_circle: if True, plot the minimum enclosing circle

    Returns:
    - ax: the axis with the sample holder plotted.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if is_min_circle:
        sampleholder.update_min_circle()
        center = sampleholder.center
        radius = sampleholder.radius
        circle = plt.Circle(
            center, radius, color="r", fill=False, linewidth=4, alpha=0.5, zorder=-100
        )
        ax.add_artist(circle)
        ax.set(
            xlim=(center[0] - 1.1 * radius, center[0] + 1.1 * radius),
            ylim=(center[1] - 1.1 * radius, center[1] + 1.1 * radius),
        )
        ax.set_aspect("equal", "box")
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

        if is_plot_contour:
            ax.fill(x_contour_new, y_contour_new, edgecolor=None)
        if is_plot_hull:
            ax.fill(x_hull_new, y_hull_new, edgecolor=None)

        if (is_plot_contour) and (not is_only_new):
            ax.plot(
                np.append(x_contour_original, x_contour_original[0]),
                np.append(y_contour_original, y_contour_original[0]),
                linestyle="--",
                color=np.array(stylesheet["contours_kwargs"]["color"])[::-1] / 255,
            )
        if is_plot_hull and (not is_only_new):
            ax.plot(
                np.append(x_hull_original, x_hull_original[0]),
                np.append(y_hull_original, y_hull_original[0]),
                linestyle="--",
                color=np.array(stylesheet["hulls_kwargs"]["color"])[::-1] / 255,
            )
        ax.invert_yaxis()
        # add the id of the sample at the position of the sample
        if sample.is_relocated:
            text_position = sample.position_new

            # draw a link between the original position and the new position
            if is_relocation_arrow:
                ax.arrow(
                    sample.position_original[0],
                    sample.position_original[1],
                    sample.position_new[0] - sample.position_original[0],
                    sample.position_new[1] - sample.position_original[1],
                    head_width=10,
                    head_length=10,
                    fc="gray",
                    ec="gray",
                    zorder=1000,
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


def visualize_vertices_list(
    vertices_list: list, ax=None, is_fill_polygon=True, is_plot_convex_hull=True
):
    """
    visualize the vertices_list

    Args:
    - vertices_list: list of vertices, each vertices is a (N, 2) numpy array, dtype=int32

    Keyword Args:
    - ax: matplotlib axis object, if None, create a new figure
    - is_fill_polygon: bool, whether to fill the polygon
    - is_plot_convex_hull: bool, whether to plot the convex hull of all points

    """
    if ax is None:
        fig, ax = plt.subplots()

    for i, vertices in enumerate(vertices_list):
        # for each vertices, we create a polygon for it and plot the polygon
        polygon = Polygon(vertices)
        x, y = polygon.exterior.xy
        if is_fill_polygon:
            ax.fill(x, y, edgecolor="black", fill=is_fill_polygon)

        if is_plot_convex_hull:
            # finished plotting above, everything below is just to plot the convex hull
            to_add_points = np.column_stack((x, y)).astype(np.int32)
            if i == 0:
                points = to_add_points
            else:
                points = np.append(points, to_add_points, axis=0)
    if is_plot_convex_hull:
        # This is to plot the convex hull of all the points in the vertices_list
        convex_hull = cv2.convexHull(points)
        # Convert the hull to a 2D array for plotting
        hull_points = convex_hull[:, 0, :]  # cv2.convexHull adds extra dimension
        hull_points = np.append(
            hull_points, [hull_points[0]], axis=0
        )  # Close the polygon
        ax.plot(hull_points[:, 0], hull_points[:, 1], "g-")

    # ax setting
    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
    return ax


def visualize_vertices(vertices: np.ndarray, ax=None, is_fill_polygon=True):
    """
    visualize the vertices
    """
    if ax is None:
        fig, ax = plt.subplots()

    # convert to polygon for a better visualization (since it arrange the vertices in the polygon automatically)
    polygon = Polygon(vertices)
    x, y = polygon.exterior.xy
    ax.fill(x, y, edgecolor="black", fill=is_fill_polygon)

    # ax settings
    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
    return ax
