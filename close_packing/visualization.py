# visualize vertices_list
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import cv2


def visualize_vertices_list(vertices_list: list, ax=None):
    """
    visualize the vertices_list
    """
    if ax is None:
        fig, ax = plt.subplots()

    for i, vertices in enumerate(vertices_list):
        polygon = Polygon(vertices)
        x, y = polygon.exterior.xy
        ax.fill(x, y, edgecolor="black", fill=True)
        to_add_points = np.column_stack((x, y)).astype(np.int32)
        if i == 0:
            points = to_add_points
        else:
            points = np.append(points, to_add_points, axis=0)
    hull = cv2.convexHull(points)
    # Convert the hull to a 2D array for plotting
    hull_points = hull[:, 0, :]  # cv2.convexHull adds extra dimension
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the polygon
    ax.plot(hull_points[:, 0], hull_points[:, 1], "g-")

    return ax
