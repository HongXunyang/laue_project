""" 
This is a package focusing on converting sample to CAD files

__version__ = 1.0
__author__ = "Xunyang Hong"

updated on 26th Oct. 2024
"""

import numpy as np
import os, trimesh
from classes import Sample, FunctionalSampleHolder, Contour
from close_packing import sampleholder2vertices_list
from shapely.geometry import Polygon
from config import config, physical_size


def vertices_list_to_cad(
    vertices_list,
    folder_path=None,
    filename=None,
    thickness=None,
):
    """
    Convert a list of vertices to a DXF file.

    Args:
    - vertices_list: list of np.ndarray, each element is a (Nx2) numpy array, dtype=np.float32

    Keyword Args:
    - folder_path: str, the folder to save the CAD file, default is defined in the config dictionary `config/config.py` file.
    - filename: str, the name of the CAD file, default is defined in the config dictionary `config/config.py` file.
    - thickness: float, the thickness of the sample, default is defined in the `physical_size` dictionary in the `config/config.py` file.

    ----------------
    # Sidenote
    Modify the commented code to save every single sample as a separate STL file.
    """
    # check folder_path and filename
    if folder_path is None:
        folder_path = config["temporary_output_folder"]
    if filename is None:
        filename = config["samples_cad_filename"]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, filename)

    thickness = (
        thickness if thickness is not None else physical_size["sample_thickness"]
    )
    meshes = []

    for idx, poly_points in enumerate(vertices_list):
        mesh = _extrude_polygon(poly_points, height=thickness)
        meshes.append(mesh)

        # Optionally, save each mesh as an individual STL file
        # mesh.export(os.path.join(folder_path, f"polygon_{idx}.stl"))

    # Combine all meshes into one
    combined_mesh = trimesh.util.concatenate(meshes)

    # Export the combined mesh as an STL file
    combined_mesh.export(path)


def sampleholder_to_cad(
    sampleholder: FunctionalSampleHolder,
    folder_path=None,
    filename=None,
    radius_multiplier: float = None,
):
    """
    Convert a sampleholder (FunctionalSampleHolder object) to a CAD file.

    Args:
    - `sampleholder`: FunctionalSampleHolder, the sample holder object. defined in the classes/class_sampleholder.py file.

    Keyword Args:
    - `folder_path`: str, the folder to save the CAD file, default is defined in the config dictionary `config/config.py` file.
    - `filename`: str, the name of the CAD file, default is defined in the config dictionary `config/config.py` file.
    - `radius_multiplier`: float, the multiplier to adjust the radius of the sample holder. For example, 1.2 means the radius is 20% larger than the minimum enclosing circle of all samples. Default is defined in the `physical_size` dictionary in the `config/config.py` file.
    """
    # check folder_path and filename
    if folder_path is None:
        folder_path = config["temporary_output_folder"]
    if filename is None:
        filename = config["sampleholder_cad_filename"]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, filename)

    # Define the radius multiplier.
    # The default value can be found in the `physical_size` dictonary in `config/config.py`
    radius_multiplier = (
        radius_multiplier
        if radius_multiplier is not None
        else physical_size["sampleholder_radius_multiplier"]
    )
    sample_thickness = (
        sampleholder.sample_thickness
        if sampleholder.sample_thickness is not None
        else 30
    )
    sampleholder_thickness = (
        sampleholder.thickness if sampleholder.thickness is not None else 60
    )
    if sampleholder.shape == "rectangle":
        # Define the holder rectangle
        holder_rectangle = _size_to_rectangle(sampleholder.size)
        # Extrude the holder rectangle to create a cuboid
        holder_thickness = 60.0  # Adjust as needed
        holder_mesh = _extrude_polygon(
            holder_rectangle, height=sampleholder_thickness
        )  # convert it to a 3D shape by extruding the rectangle
        engrave_meshes = []

        vertices_list = sampleholder2vertices_list(sampleholder)

        for idx, poly_points in enumerate(vertices_list):
            engrave_mesh = _extrude_polygon(poly_points, height=sample_thickness)
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
        # Export the final engraved holder as an STL file
        final_mesh.export(path)

    elif sampleholder.shape == "circle":
        radius = sampleholder.radius * radius_multiplier
        center = sampleholder.center
        # Create a mesh of a cylinder along Z centered at the origin.
        holder_mesh = trimesh.creation.cylinder(
            radius=radius, height=sampleholder_thickness, sections=64
        )
        translation = np.append(center, 0) + np.array(
            [0, 0, sampleholder_thickness / 2]
        )
        holder_mesh.apply_translation(translation)
        engrave_meshes = []

        vertices_list = sampleholder2vertices_list(sampleholder)

        for idx, poly_points in enumerate(vertices_list):
            engrave_mesh = _extrude_polygon(poly_points, height=sample_thickness)
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
        # Export the final engraved holder as an STL file
        final_mesh.export(path)
    print(f"Exported the sample holder to {path} in STL format.")


def sampleholder_dict_to_cad(
    sampleholder_dict: dict,
    folder_path=None,
    filename=None,
    radius_multiplier: float = None,
):
    """
    Convert a sample holder dictionary to a CAD file.

    Args:
    - `sampleholder_dict`: dict, a dictionary containing the sample holder information.

    Keyword Args:
    - `folder_path`: str, the folder to save the CAD file, default is defined in the config dictionary `config/config.py` file.
    - `filename`: str, the name of the CAD file, default is defined in the config dictionary `config/config.py` file.
    - `radius_multiplier`: float, the multiplier to adjust the radius of the sample holder. For example, 1.2 means the radius is 20% larger than the minimum enclosing circle of all samples. Default is defined in the `physical_size` dictionary in the `config/config.py` file.

    ----------------
    # Sidenote
    The dictionary of the sampleholder is usually saved as a Json file. Please refer to the `utils.save_sampleholder()` for more details on the directory structure.
    """
    # check folder_path and filename
    if folder_path is None:
        folder_path = config["temporary_output_folder"]
    if filename is None:
        filename = config["sampleholder_dict_filename"]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, filename)
    radius_multiplier = (
        radius_multiplier
        if radius_multiplier is not None
        else physical_size["sampleholder_radius_multiplier"]
    )

    sample_thickness = (
        sampleholder_dict["sample_thickness"]
        if "sample_thickness" in sampleholder_dict
        else 30
    )
    sampleholder_thickness = sampleholder_dict["thickness"]
    if sampleholder_dict["shape"] == "rectangle":
        # Define the holder rectangle
        holder_rectangle = _size_to_rectangle(sampleholder_dict["size"])
        # Extrude the holder rectangle to create a cuboid
        holder_mesh = _extrude_polygon(
            holder_rectangle, height=sampleholder_thickness
        )  # convert it to a 3D shape by extruding the rectangle
        engrave_meshes = []

        vertices_list = sampleholder2vertices_list(sampleholder_dict)

        for idx, poly_points in enumerate(vertices_list):
            engrave_mesh = _extrude_polygon(poly_points, height=sample_thickness)
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
        # Export the final engraved holder as an STL file
        final_mesh.export(path)

    else:
        radius = sampleholder_dict["radius"] * radius_multiplier
        center = np.array(sampleholder_dict["center"])
        # Create a mesh of a cylinder along Z centered at the origin.
        holder_mesh = trimesh.creation.cylinder(
            radius=radius, height=sampleholder_thickness, sections=64
        )
        translation = np.append(center, 0) + np.array(
            [0, 0, sampleholder_thickness / 2]
        )
        holder_mesh.apply_translation(translation)
        engrave_meshes = []

        vertices_list = sampleholder_dict["vertices_list"]

        for idx, poly_points in enumerate(vertices_list):
            engrave_mesh = _extrude_polygon(poly_points, height=sample_thickness)
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
        # Export the final engraved holder as an STL file
        final_mesh.export(path)
    print(f"done generating STL file at {path}")


def _extrude_polygon(polygon_points, height):
    # Create a shapely polygon from the points
    poly = Polygon(polygon_points)

    # Ensure the polygon is valid
    if not poly.is_valid:
        poly = poly.buffer(0)  # Fixes self-intersecting polygons

    # Extrude the polygon to create a 3D mesh
    mesh = trimesh.creation.extrude_polygon(poly, height)
    return mesh


def _size_to_rectangle(size):
    # convert a 2x1 tuple to a rectangle points
    width, height = size

    rectangle = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    return rectangle
