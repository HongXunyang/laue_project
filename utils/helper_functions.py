import numpy as np
import json
from shapely.geometry import Polygon
from classes import Contour, FunctionalSampleHolder, Sample


def sampleholder2polygons(sampleholder: FunctionalSampleHolder):
    """
    given a sample holder, return a list of polygons
    """
    samples_list = sampleholder.samples_list
    return [sample.contour_new.polygon for sample in samples_list]


def sampleholder2vertices_list(sampleholder: FunctionalSampleHolder):
    """
    given a sample holder, return a list of vertices. A vertices is a (N, 2) numpy array, dtype=int32
    """
    samples_list = sampleholder.samples_list
    return [sample.contour_new.vertices for sample in samples_list]


def sample2polygon(sample: Sample):
    """given a sample, return the polygon"""
    return sample.contour_new.polygon


def is_two_polygons_overlap(polygon1: Polygon, polygon2: Polygon):
    return polygon1.intersects(polygon2)


def is_polygon_overlap_with_polygons(polygon: Polygon, polygons: list):
    for polygon_ in polygons:
        if is_two_polygons_overlap(polygon, polygon_):
            return True
    return False


def vertices_area(vertices: np.ndarray):
    """
    given a vertices, return the area of the polygon
    """
    return Polygon(vertices.tolist()).area


def update_sampleholder(sampleholder: FunctionalSampleHolder, new_vertices_list: list):
    """
    update the sampleholder with the new vertices configuration. This will update the samples_list in the sampleholder.

    P.S. This is different from sampleholder.update() which only updates the sampleholder's parameters based on the existing samples_list.
    """
    old_vertices_list = sampleholder2vertices_list(sampleholder)
    _update_sampleholder(sampleholder, old_vertices_list, new_vertices_list)
    return sampleholder


def _update_sampleholder(
    sampleholder: FunctionalSampleHolder, old_vertices_list, new_vertices_list: list
):
    """
    update the sampleholder with the new configuration
    """
    # first update the samples in the sample holder
    for i, vertices in enumerate(new_vertices_list):
        # determine the translation offset between the original and new vertices
        translation = _get_translation(old_vertices_list[i], new_vertices_list[i])
        sample = sampleholder.samples_list[i]
        # update the sample.position_original based on the old_vertices
        sample.position_original = np.mean(old_vertices_list[i], axis=0)
        # update the sample.position_new before applying sample.relocate()
        sample.position_new = sample.position_original + translation
        sample.relocate()

    # second, update the sampleholder
    sampleholder.update()


def _get_translation(old_vertices: np.ndarray, new_vertices: np.ndarray) -> np.ndarray:
    """get the translation vector by comparing the center of mass of the old and new vertices"""

    old_center = np.mean(old_vertices, axis=0)
    new_center = np.mean(new_vertices, axis=0)

    return new_center - old_center


def save_sampleholder(sampleholder, folder_path, filename):
    """
    save the data stored in the sampleholder in the form of a dictionary to a json file
    """
    sampleholder.update()
    name: str = sampleholder.name
    shape: str = sampleholder.shape if (sampleholder.shape is not None) else "None"
    size: list[float, float] = (
        sampleholder.size.tolist() if (sampleholder.size is not None) else "None"
    )
    radius: float = sampleholder.radius if (sampleholder.radius is not None) else "None"
    thickness: float = (
        sampleholder.thickness if (sampleholder.thickness is not None) else "None"
    )
    center: list[float, float] = (
        sampleholder.center.tolist() if (sampleholder.center is not None) else "None"
    )
    samples_area: float = (
        sampleholder.samples_area if (sampleholder.samples_area is not None) else "None"
    )
    ratio: float = sampleholder.ratio if (sampleholder.ratio is not None) else "None"
    convex_hull: list[list[list[float, float]]] = (
        sampleholder.convex_hull.tolist()
        if (sampleholder.convex_hull is not None)
        else "None"
    )
    vertices_list: list[list[list[float, float]]] = (
        sampleholder.vertices_list.tolist()
        if (sampleholder.vertices_list is not None)
        else "None"
    )

    sampleholder_dict = dict(
        name=name,
        shape=shape,
        size=size,
        radius=radius,
        thickness=thickness,
        center=center,
        samples_area=samples_area,
        ratio=ratio,
        convex_hull=convex_hull,
        vertices_list=vertices_list,
    )

    with open(f"{folder_path}+{filename}", "w") as f:
        json.dump(sampleholder_dict, f)

    print(f"Sampleholder data saved to {folder_path}+{filename}")

    return sampleholder_dict


def load_sampleholder(folder_path, filename):
    """
    load the sampleholder data from the json file
    """
    with open(f"{folder_path}+{filename}", "r") as f:
        sampleholder_dict = json.load(f)

    sampleholder = FunctionalSampleHolder()
    sampleholder.name = sampleholder_dict["name"]
    sampleholder.shape = sampleholder_dict["shape"]
    sampleholder.size = sampleholder_dict["size"]
    sampleholder.radius = sampleholder_dict["radius"]
    sampleholder.thickness = sampleholder_dict["thickness"]
    sampleholder.center = sampleholder_dict["center"]
    sampleholder.samples_area = sampleholder_dict["samples_area"]
    sampleholder.ratio = sampleholder_dict["ratio"]
    sampleholder.convex_hull = np.array(
        sampleholder_dict["convex_hull"], dtype=np.float32
    )
    vertices_list = [
        np.array(vertices, dtype=np.float32)
        for vertices in sampleholder_dict["vertices_list"]
    ]
    sampleholder.vertices_list = vertices_list

    # rebuild the samples_list
    print("XUN-WARNING - Currently rebuidling samples_list not supported...")
    return sampleholder
