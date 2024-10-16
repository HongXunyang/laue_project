import cv2
import numpy as np
import matplotlib.pyplot as plt
from packages import (
    unify_background,
    remove_stripes,
    contours2polygons,
    image2contours,
    visualize_contours,
)


stripes_vectors = [
    np.array([95, 86, 167]),
    np.array([57, 48, 139]),
    np.array([72, 66, 137]),
]
target_background_vector = np.array([202, 209, 206])
background_vectors = [
    np.array([202, 209, 206]),
    np.array([190, 201, 199]),
    np.array([182, 185, 183]),
]


# Callback function to get the BGR value of the clicked pixel
def get_bgr_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # If left mouse button is clicked
        # Get the BGR value at the (x, y) coordinate
        bgr_value = image[y, x]
        # Extract the individual blue, green, red values
        blue = bgr_value[0]
        green = bgr_value[1]
        red = bgr_value[2]
        print(f"BGR Value at ({x}, {y}): Blue={blue}, Green={green}, Red={red}")
        # Display the value on the image as text (optional)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            f"BGR:({blue},{green},{red})",
            (x, y),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Resized Window", image)


# Load image
original_image = cv2.imread("../images/fake_holder_with_samples.jpg")
rows, columns, channels = original_image.shape
original_image = original_image[
    int(0.15 * rows) : int(0.435 * rows), int(0.1 * columns) : int(0.9 * columns)
]
if False:
    cv2.namedWindow("Resized Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized Window", 400, 600)
    cv2.setMouseCallback("Resized Window", get_bgr_value)
    # Display the image and wait for a key press
    while True:
        cv2.imshow("Resized Window", image)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing the 'Esc' key
            break
    cv2.imshow("Resized Window", image)

orginal_image = cv2.resize(
    original_image, (rows // 5, columns // 5), interpolation=cv2.INTER_AREA
)
rows, columns, channels = orginal_image.shape

contours, approximated_contours, hulls = image2contours(
    original_image,
    gaussian_window=(7, 7),
    stripes_vectors=stripes_vectors,
    background_vectors=background_vectors,
    epsilon=2.5,
    lowercut=100,
)


# Visualize the contours
contours_kwargs = {
    "color": (0, 0, 255),
    "thickness": 4,
}
hulls_kwargs = {
    "color": (0, 255, 0),
    "thickness": 3,
}
image_to_visualize = visualize_contours(
    original_image,
    approximated_contours,
    hulls,
    contours_kwargs=contours_kwargs,
    hulls_kwargs=hulls_kwargs,
)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 800, 500)
cv2.imshow("result", image_to_visualize)
cv2.waitKey(0)

print("done")
