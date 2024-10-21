""" 
This is a package focusing on converting sample to DXF file
"""

import numpy as np
import ezdxf
from classes import Sample, FunctionalSampleHolder, Contour
from shapely.geometry import Polygon
import trimesh


def vertices_to_list(vertices: np.ndarray):
    pass


def vertices_list_to_cad(
    vertices_list,
    cad_file,
    cad_folder: str = "data/",
    is_3d=False,
    thickness: float = 30,
    is_print=True,
):
    """
    Convert a list of vertices to a DXF file.

    Args:
    vertices_list: list of np.ndarray, each element is a (Nx2) numpy array, dtype=np.float32
    cad_file: str, the name of the cad file

    Keyword Args:
    cad_folder: str, the folder to save the DXF file, "data/" by default
    is_3d: bool, whether to convert the polygons to 3D shapes. If false, the polygons will be 2D.
    thickness: float, the thickness of the 3D shape.
    is_print: bool, whether to print the message after the DXF file is created.

    """
    # create a new DXF document
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    if not is_3d:
        polygons = [None] * len(vertices_list)
        for i, vertices in enumerate(vertices_list):
            polygon = vertices.tolist()

            # make sure the polygon is closed
            if polygon[0] != polygon[-1]:
                polygon.append(polygon[0])

            polygons[i] = polygon

            msp.add_lwpolyline(polygon)

        doc.saveas(cad_folder + cad_file)
        if is_print:
            print(f"[2D polygons] DXF file at {cad_file} has been created.")

    else:
        # here we convert the polygons to a 3D shape, with certain thickness.
        meshes = []

        for idx, poly_points in enumerate(vertices_list):
            mesh = _extrude_polygon(poly_points, height=thickness)
            meshes.append(mesh)

            # Optionally, save each mesh as an individual STL file
            mesh.export(cad_folder + f"polygon_{idx}.stl")

        # Combine all meshes into one
        combined_mesh = trimesh.util.concatenate(meshes)

        # Export the combined mesh as an STL file
        combined_mesh.export(cad_folder + cad_file)


def contour_to_dxf(contour: Contour, dxf_file):
    pass


def sample_to_dxf(sample: Sample, dxf_file):
    pass


def sample_list_to_dxf(sample_list: list, dxf_file):
    pass


def sampleholder_to_dxf(sampleholder: FunctionalSampleHolder, dxf_file):
    pass


def _extrude_polygon(polygon_points, height):
    # Create a shapely polygon from the points
    poly = Polygon(polygon_points)

    # Ensure the polygon is valid
    if not poly.is_valid:
        poly = poly.buffer(0)  # Fixes self-intersecting polygons

    # Extrude the polygon to create a 3D mesh
    mesh = trimesh.creation.extrude_polygon(poly, height)

    return mesh
