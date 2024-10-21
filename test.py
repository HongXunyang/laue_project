import numpy as np
from shapely.geometry import Polygon
import trimesh


def extrude_polygon(polygon_points, height):
    # Create a shapely polygon from the points
    poly = Polygon(polygon_points)

    # Ensure the polygon is valid
    if not poly.is_valid:
        poly = poly.buffer(0)  # Fixes self-intersecting polygons

    # Extrude the polygon to create a 3D mesh
    mesh = trimesh.creation.extrude_polygon(poly, height)

    return mesh


# Example list of polygons
polygons = [
    np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # Square
    np.array([[2, 2], [3, 2], [2.5, 3]]),  # Triangle
    # Add your polygons here
]

# Thickness you want to add (along the Z-axis)
thickness = 5.0  # Adjust as needed

# List to store the meshes
meshes = []

for idx, poly_points in enumerate(polygons):
    mesh = extrude_polygon(poly_points, height=thickness)
    meshes.append(mesh)

    # Optionally, save each mesh as an individual STL file
    mesh.export(f"polygon_{idx}.stl")


# Combine all meshes into one
combined_mesh = trimesh.util.concatenate(meshes)

# Export the combined mesh as an STL file
combined_mesh.export("combined_polygons.stl")
