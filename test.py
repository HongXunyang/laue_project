import numpy as np
from shapely.geometry import Polygon
import trimesh


def extrude_polygon(polygon_points, height, direction="up"):
    """
    Extrude a 2D polygon into a 3D mesh with the given height.

    Parameters:
    - polygon_points: (Nx2) numpy array of polygon vertices.
    - height: Thickness to extrude.
    - direction: 'up' or 'down' extrusion along Z-axis.

    Returns:
    - extruded_mesh: trimesh.Trimesh object.
    """
    poly = Polygon(polygon_points)

    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.is_valid:
            raise ValueError("Invalid polygon. Cannot be fixed with buffer(0).")

    # Extrude the polygon
    extruded_mesh = trimesh.creation.extrude_polygon(poly, height)

    if direction == "down":
        # Invert the extrusion along Z-axis
        extruded_mesh.apply_translation([0, 0, -height])

    return extruded_mesh


def main():
    # Define the holder rectangle (example: 100x100 units)
    holder_rectangle = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])

    # Extrude the holder rectangle to create a cuboid
    holder_thickness = 60.0  # Adjust as needed
    holder_mesh = extrude_polygon(
        holder_rectangle, height=holder_thickness, direction="up"
    )

    # Example list of sample polygons
    sample_polygons = [
        np.array([[10, 10], [30, 10], [30, 30], [10, 30]]),  # Square
        np.array([[40, 40], [60, 40], [50, 60]]),  # Triangle
        # Add more polygons as needed
    ]

    # Thickness for engraving
    engrave_thickness = 30.0  # Adjust as needed

    # List to store extruded sample meshes
    engrave_meshes = []

    for idx, poly_points in enumerate(sample_polygons):
        engrave_mesh = extrude_polygon(
            poly_points, height=engrave_thickness, direction="up"
        )
        engrave_meshes.append(engrave_mesh)

        # Optional: Save individual engrave meshes for debugging
        # engrave_mesh.export(f'engrave_{idx}.stl')

    # Combine all engrave meshes into one
    combined_engrave_mesh = trimesh.util.concatenate(engrave_meshes)

    # Perform boolean subtraction to engrave the holder
    final_mesh = holder_mesh.difference(combined_engrave_mesh)

    # Check if the boolean operation was successful
    if final_mesh is None or final_mesh.is_empty:
        raise ValueError(
            "Boolean operation failed. Ensure that OpenSCAD is installed and accessible."
        )

    # Optional: Visualize the final mesh
    # final_mesh.show()

    # Export the final engraved holder as an STL file
    final_mesh.export("engraved_holder.stl")
    print("Engraved holder STL exported successfully as 'engraved_holder.stl'.")


if __name__ == "__main__":
    main()
