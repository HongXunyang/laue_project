from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import time, cv2, json
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import animate_config_evolution, save_sampleholder, visualize_vertices_list
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from config.config import (
    batch_optimization_kwargs,
    tests_config,
    image2contours_kwargs,
)
from close_packing import batch_optimization, optimization

STEP_CONTROL = dict(
    test=True, contour_finding=True, close_packing=True, convert_to_cad=True
)


# ----------- image pre-processing ----------- #
if STEP_CONTROL["contour_finding"] or STEP_CONTROL["test"]:
    start_time = time.time()
    stripes_vectors, background_vectors, target_background_vector = (
        tests_config["stripes_vectors"],
        tests_config["background_vectors"],
        tests_config["target_background_vector"],
    )
    # Load image
    image = cv2.imread(tests_config["test_image_path"])
    rows, columns, channels = image.shape

    # ----------- end of image pre-processing ----------- #
    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # ---------------- contour finding ------------------ #

    # Find the contours of the samples in the image
    # by default, the output images will be saved in the `temporary_output/` folder.
    contours, approximated_contours, hulls, _ = image2contours(
        image,
        stripes_vectors=stripes_vectors,
        background_vectors=background_vectors,
        **image2contours_kwargs,
    )
    end_time = time.time()
    print(f"image processed time: {end_time - start_time} seconds\n")

    # create samples objects and sample holder object
    samples_list = generate_sample_objects(approximated_contours, hulls)
    sampleholder = generate_sampleholder_object(samples_list)

    # save initial configuration
    sampleholder.update()
    save_sampleholder(sampleholder, filename="initial_sampleholder.json")

# load the json file


import json
from shapely.geometry import Polygon

with open("initial_sampleholder.json", "r") as f:
    sampleholder_dict = json.load(f)

vertices_list = sampleholder_dict["vertices_list"]
""" 
the coordinates of the vertices of each polygon are stored in the `vertices_list` variable. There are 23 polygons. To access them:
- the first polygon: `vertices_list[0]`
- the last polygon: `vertices_list[22]`

when you do this, you will get a list of 2d coordinates:
vertices_list[0] = [[363.0, 734.0], [339.0, 762.0], [284.0, 800.0], [262.0, 804.0], [246.0, 793.0], [237.0, 781.0], [255.0, 737.0], [273.0, 721.0], [326.0, 717.0]]

- vertices_list[0][0] = [363.0, 734.0], this is the coordinate of the first vertex of the first polygon
- vertices_list[0][1] = [339.0, 762.0], this is the coordinate of the second vertex of the first polygon
- etc.

to convert them into shapely Polygon, you need to:
1. from shapely.geometry import Polygon
2. poly0 = Polygon(vertices_list[0])
3. poly1 = Polygon(vertices_list[1])
3. ...
"""


poly0 = Polygon(vertices_list[0])
# visualize polygon
fig, ax = plt.subplots()
x, y = poly0.exterior.xy
ax.plot(x, y, color="b")


print(poly0)
fig, ax = plt.subplots()
visualize_vertices_list(vertices_list, ax=ax)
plt.show()
