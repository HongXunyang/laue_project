import cv2
import numpy as np
import json


def _min_circle_radius(vertices_list):
    """
    calculate the center and radius of the minimum enclosing circle of the given vertices list

    args:
    - vertices_list: the vertices list of the polygon.

    returns:
    - center: the center of the minimum enclosing circle
    - radius: the radius of the minimum enclosing circle
    """
    # Extract all points
    points = np.array([point for vertices in vertices_list for point in vertices])
    points = points.astype(np.float32)
    convex_hull = cv2.convexHull(points)
    # calculate the minimum enclosing circle of the convex hull
    center, radius = cv2.minEnclosingCircle(convex_hull)
    return center, radius


# read json file in temporary_output/sampleholder.json
with open("temporary_output/sampleholder.json", "r") as f:
    sampleholder_dict = json.load(f)

# get the vertices_list
vertices_list = sampleholder_dict["vertices_list"]

# calculate the radius of the minimum enclosing circle
center, radius = _min_circle_radius(vertices_list)
print(center, radius)
