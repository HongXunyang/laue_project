import shapely.geometry as geom
import shapely.affinity
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np

# Define shapes (convex polygons)
shapes = [
    geom.Polygon([(0, 0), (2, 0), (1, 1.732)]),
    geom.Polygon([(0, 0), (1, 0), (0.5, 0.866)]),
    # Add more shapes
]

# First-Fit-Decreasing: Sort shapes by decreasing area
shapes.sort(key=lambda s: s.area, reverse=True)

# Initial container size
container_width = container_height = 10  # Adjust as needed
container = geom.Polygon(
    [
        (0, 0),
        (container_width, 0),
        (container_width, container_height),
        (0, container_height),
    ]
)


def minkowski_sum(A, B):
    # Reflect shape B about the origin
    B_reflected = shapely.affinity.scale(B, xfact=-1, yfact=-1, origin=(0, 0))
    # Compute Minkowski sum
    coords = []
    for a in A.exterior.coords:
        for b in B_reflected.exterior.coords:
            coords.append((a[0] + b[0], a[1] + b[1]))
    return geom.Polygon(coords).convex_hull


def compute_feasible_region(container, placed_shapes, current_shape):
    feasible_region = container

    for placed_shape in placed_shapes:
        nfp = minkowski_sum(placed_shape, current_shape)
        feasible_region = feasible_region.difference(nfp)

    return feasible_region


def place_shapes(shapes, container):
    placed_shapes = []
    for shape in shapes:
        feasible_region = compute_feasible_region(container, placed_shapes, shape)

        if feasible_region.is_empty or feasible_region.area == 0:
            print("Unable to place shape:", shape)
            return None  # Placement failed

        # Choose a point to place the shape (e.g., bottom-leftmost point)
        minx, miny, maxx, maxy = feasible_region.bounds
        position = (minx, miny)

        # Move the shape to the new position
        moved_shape = shapely.affinity.translate(
            shape, xoff=position[0], yoff=position[1]
        )

        placed_shapes.append(moved_shape)

    return placed_shapes


# Try placing shapes
placed_shapes = place_shapes(shapes, container)


# Visualization
def plot_shapes(shapes, container):
    fig, ax = plt.subplots()
    x, y = container.exterior.xy
    ax.plot(x, y, color="black")

    for shape in shapes:
        x, y = shape.exterior.xy
        ax.fill(x, y, alpha=0.5, fc="blue", ec="black")

    ax.set_aspect("equal")
    plt.show()


if placed_shapes:
    plot_shapes(placed_shapes, container)
else:
    print("Failed to place all shapes.")
