import time, cv2, os
from matplotlib import pyplot as plt
from contour_finding import (
    image2contours,
    generate_sample_objects,
    generate_sampleholder_object,
)
from utils import animate_config_evolution, visualize_contours
from to_cad import vertices_list_to_cad, sampleholder_to_cad
from config.config import (
    batch_optimization_kwargs,
    tests_config,
    image2contours_kwargs,
)
from close_packing import batch_optimization, optimization
from utils.report_generator import ReportGenerator


STEP_CONTROL = dict(
    test=False, contour_finding=True, close_packing=True, convert_to_cad=True
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
    contours = [sample.contour_original.contour for sample in sampleholder.samples_list]
    hulls = [sample.contour_original.hull for sample in sampleholder.samples_list]
    visualize_contours(
        image,
        contours,
        hulls,
        is_plot=False,
        is_output_image=True,
    )
# ----------- end of contour finding ----------- #
# ----------------- Optimization --------------- #

start_time = time.time()
best_configuration, area_list, sorted_indices, area_evolution_list = batch_optimization(
    sampleholder,
    **batch_optimization_kwargs,
)
end_time = time.time()
print(f"optimization time: {end_time - start_time} seconds\n")

# ----------- end of optimization ----------- #
# # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ----------- convert Samples to CAD --------- #
vertices_list = sampleholder.vertices_list
# Save the configuration of the samples to a CAD (STL) file
# by default, the output CAD files will be saved in the `temporary_output/` folder.
vertices_list_to_cad(
    vertices_list,
)
# update sampleholder object
sampleholder.shape = "circle"
sampleholder.update()

# Save the engraved sampleholder to a CAD (STL) file
# by default, the output CAD files will be saved in the `temporary_output/` folder.
sampleholder_to_cad(
    sampleholder,
)
# ----------- end  ----------- #

# Generate report after all processing is done
try:
    generator = ReportGenerator()
    optimization_results = {
        "final_ratio": sampleholder.ratio if "sampleholder" in locals() else None,
        "iterations": (
            len(area_evolution_list) if "area_evolution_list" in locals() else None
        ),
        "computation_time": (
            end_time - start_time
            if "end_time" in locals() and "start_time" in locals()
            else None
        ),
    }

    report_path = generator.generate_report(
        sampleholder if "sampleholder" in locals() else None, optimization_results
    )
    print(f"\nReport generated successfully at: {report_path}")

    # Optionally open the report
    import webbrowser

    webbrowser.open(f"file://{os.path.abspath(report_path)}")

except Exception as e:
    print(f"\nError generating report: {str(e)}")

plt.show()
