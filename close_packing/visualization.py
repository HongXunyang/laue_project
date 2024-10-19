import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import cv2


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
