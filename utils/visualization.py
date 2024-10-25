import numpy as np
import json
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from classes import _remove_background_contour
from config.config import (
    plot_area_evolution_kwargs,
    plot_ratio_evolution_kwargs,
    config,
)

with open("config/stylesheet.json", "r") as json_file:
    stylesheet = json.load(json_file)


def visualize_sampleholder(
    sampleholder,
    ax=None,
    is_plot_contour=False,
    is_plot_hull=True,
    is_min_circle=True,
):
    """
    This method visualizes the sample holder.
    - Plot the original contours with dashed lines
    - Plot the new contours after reorientation or relocation with solid lines
    - indicate the movement of the samples with arrows

    Args:
    sampleholder: a FunctionalSampleHolder object

    Keyword arguments:
    - ax: the axis to plot the sample holder.
    - is_plot_contour: if True, plot the contours of each samples
    - is_plot_hull: if True, plot the hulls of each samples
    - is_fill_new_polygon: if True, fill the new polygon with color
    - is_min_circle: if True, plot the minimum enclosing circle

    Returns:
    - ax: the axis with the sample holder plotted.
    """
    if ax is None:
        fig, ax = plt.subplots()

    sampleholder.update()
    # plot the minimum enclosing circle
    if is_min_circle:
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

    # plot samples
    for i, sample in enumerate(sampleholder.samples_list):
        contour_new = sample.contour_new.contour
        hull_new = sample.contour_new.hull

        # plot the new contour and hull using solid line
        x_contour_new = [point[0][0] for point in contour_new]
        y_contour_new = [point[0][1] for point in contour_new]
        x_hull_new = [point[0][0] for point in hull_new]
        y_hull_new = [point[0][1] for point in hull_new]

        if is_plot_contour:
            ax.fill(x_contour_new, y_contour_new, edgecolor=None)
        if is_plot_hull:
            ax.fill(x_hull_new, y_hull_new, edgecolor=None)
        ax.invert_yaxis()
        # add the id of the sample at the position of the sample
        if sample.is_relocated:
            text_position = sample.position_new
        else:
            text_position = sample.position_original
        ax.text(
            text_position[0],
            text_position[1],
            sample.id,
            fontsize=12,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

    return ax


def visualize_sampleholder_dict(sampleholder_dict, ax=None, is_min_circle=True):
    pass
    """ 
    same as `visualize_sampleholder`, but use the sampleholder_dict instead of the sampleholder object
    """
    if ax is None:
        fig, ax = plt.subplots()

    # plot the minimum enclosing circle
    if is_min_circle:
        center = sampleholder_dict["center"]
        radius = sampleholder_dict["radius"]
        circle = plt.Circle(
            center, radius, color="r", fill=False, linewidth=4, alpha=0.5, zorder=-100
        )
        ax.add_artist(circle)
        ax.set(
            xlim=(center[0] - 1.1 * radius, center[0] + 1.1 * radius),
            ylim=(center[1] - 1.1 * radius, center[1] + 1.1 * radius),
        )
        ax.set_aspect("equal", "box")

    # plot samples
    for i, vertices in enumerate(sampleholder_dict["vertices_list"]):

        # vertices is a list of points, each point is a list of x and y coordinates
        x = [point[0] for point in vertices]
        y = [point[1] for point in vertices]

        ax.fill(x, y, edgecolor=None)
        ax.invert_yaxis()
        # add the id of the sample at the position of the sample
        text_x = np.mean(x)
        text_y = np.mean(y)
        ax.text(
            text_x,
            text_y,
            i,
            fontsize=12,
            color="black",
        )

    return ax


def visualize_contours(
    image,
    contours,
    hulls,
    is_remove_background_contour=True,
    is_plot=False,
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
            # horizontal alignment: center, vertical alignment: center
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


def visualize_area_evolution(
    sampleholder,
    area_evolution_list,
    ax_area,
    ax_ratio,
):
    sampleholder.update()
    samples_area = sampleholder.samples_area
    area_evolution_list = np.array(area_evolution_list) ** 2 * np.pi
    for i, area_evolution in enumerate(area_evolution_list):
        ax_area.plot(
            area_evolution,
            color=plot_area_evolution_kwargs["color"],
            alpha=plot_area_evolution_kwargs["alpha"],
            linewidth=plot_area_evolution_kwargs["linewidth"],
        )
        ax_area.set_ylabel(
            "Area of sampleholder", color=plot_area_evolution_kwargs["color"]
        )
        ax_area.set_xlabel("Iteration")
        ax_area.set(yticks=[])
        ax_ratio.plot(
            100 * samples_area / area_evolution,
            color=plot_ratio_evolution_kwargs["color"],
            alpha=plot_ratio_evolution_kwargs["alpha"],
            linewidth=plot_ratio_evolution_kwargs["linewidth"],
        )
        ax_ratio.set_ylabel("Ratio (%)", color=plot_ratio_evolution_kwargs["color"])
        ax_ratio.tick_params(axis="y", labelcolor=plot_ratio_evolution_kwargs["color"])
        ax_ratio.set(yticks=[0, 20, 40, 60, 80])

    # on ax_ratio, plot the largest ratio as a horizontal line
    max_ratio = np.max(100 * samples_area / np.array(area_evolution_list))
    ax_ratio.axhline(
        max_ratio,
        color=plot_ratio_evolution_kwargs["color"],
        linestyle="--",
        linewidth=1.5,
    )
    # on the top of the line, add the text of the largest ratio, on the right hand side of the line
    ax_ratio.text(
        len(area_evolution) - 1,
        max_ratio,
        f"{max_ratio:.2f}%",
        color=plot_ratio_evolution_kwargs["color"],
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
    )


# ----------------- Animation -----------------
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MatPlotPolygon


def animate_config_evolution(
    configurations,
    area_evolution,
    samples_area,
    fig=None,
    axs=None,
    is_save=False,
    filename=None,
    max_duration=20,
):
    """
    Animates the optimization process of polygon configurations alongside the area evolution.

    Parameters:
    - configurations: list of configurations, where each configuration is a list of polygons,
      and each polygon is a list of (x, y) tuples.
    - area_evolution: list of area values corresponding to each configuration.
    """
    # Set up the figure and axes
    if axs is None or fig is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax_ratio, ax_config = axs
    ax_config.set_aspect("equal")
    area_evolution = np.array(area_evolution) ** 2 * np.pi
    ratio_evolution = 100 * samples_area / np.array(area_evolution)
    # Initialize the list to store polygon patches
    polygon_patches = []

    # Create polygon patches for the initial frame
    initial_polygons = configurations[0]
    for polygon_coords in initial_polygons:
        polygon_patch = MatPlotPolygon(polygon_coords, closed=True, edgecolor="k")
        ax_config.add_patch(polygon_patch)
        polygon_patches.append(polygon_patch)

    # Set the plot limits for configuration plot
    all_x = [x for config in configurations for poly in config for x, y in poly]
    all_y = [y for config in configurations for poly in config for x, y in poly]
    ax_config.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax_config.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax_config.set_title("Configuration Evolution")

    # Initialize the area plot
    (ratio_line,) = ax_ratio.plot(
        [],
        [],
        color=plot_ratio_evolution_kwargs["color"],
        linewidth=plot_ratio_evolution_kwargs["linewidth"],
    )
    ax_ratio.set_xlim(0, len(ratio_evolution))
    ax_ratio.set_ylim(0, 80)
    ax_ratio.set_title("Ratio Evolution")
    ax_ratio.set_xlabel("Iteration")
    ax_ratio.set_ylabel("Ratio (%)")
    ax_ratio.set_yticks([0, 20, 40, 60, 80])

    # Initialize the horizontal dashed line and text annotation
    horizontal_line = ax_ratio.axhline(
        y=ratio_evolution[0], color=plot_ratio_evolution_kwargs["color"], linestyle="--"
    )
    text_annotation = ax_ratio.text(0, ratio_evolution[0], "", va="bottom", ha="center")

    def init():
        """Initialize the background of the animation."""
        ratio_line.set_data([], [])
        horizontal_line.set_ydata(ratio_evolution[0])
        text_annotation.set_text(f"{ratio_evolution[0]:.2f}%")
        return polygon_patches + [ratio_line, horizontal_line, text_annotation]

    def update(frame):
        """Update the polygons and area plot for each frame."""
        # Update configuration polygons
        polygons = configurations[frame]
        for patch, coords in zip(polygon_patches, polygons):
            patch.set_xy(coords)
        current_ratio = ratio_evolution[frame]
        horizontal_line.set_ydata(current_ratio)
        text_annotation.set_position((frame, current_ratio))
        text_annotation.set_text(f"{current_ratio:.2f}%")

        # Update area plot
        ratio_line.set_data(range(frame + 1), ratio_evolution[: frame + 1])
        return polygon_patches + [ratio_line, horizontal_line, text_annotation]

    # Create the animation
    total_frames = len(configurations) // 7
    interval = min(3, max_duration * 1000 / total_frames)  # interval in ms
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(configurations), 7),
        init_func=init,
        blit=True,
        interval=interval,
    )
    filename = filename if (filename is not None) else "config_and_area_evolution.mp4"
    if is_save:
        ani.save(
            filename,
            writer="ffmpeg",
            fps=1000 / interval,
        )
    # Display the animation
    plt.show()
