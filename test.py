import cv2
import numpy as np

# Load the image in grayscale
image_path = "../images/first_test.jpg"  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original grayscale image
cv2.imshow("Original Grayscale Image", image)

# Apply a Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Display the blurred image
cv2.imshow("Blurred Image", blurred_image)

# Apply Otsu's thresholding to separate the dark samples from the background
ret, thresh = cv2.threshold(
    blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Display the thresholded image
cv2.imshow("Thresholded Image", thresh)


# Apply morphological operations to remove small noise (e.g., stripes)
kernel = np.ones((3, 3), np.uint8)
# Remove noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Display the image after opening
cv2.imshow("After Opening", opening)


# Close small holes inside the foreground objects
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Display the image after closing
cv2.imshow("After Closing", closing)


# Find contours from the processed binary image
contours, hierarchy = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Create a copy of the original image to draw all contours
image_all_contours = cv2.imread(image_path)

# Draw all contours (before filtering)
cv2.drawContours(image_all_contours, contours, -1, (0, 0, 255), 2)

# Display the image with all contours drawn
cv2.imshow("All Contours (Before Filtering)", image_all_contours)


# Filter contours by area to exclude small contours (like stripes)
min_area_threshold = 1000  # Adjust this value based on your image
filtered_contours = [
    cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold
]

# Load the original image in color to draw the filtered contours on it
image_filtered_contours = cv2.imread(image_path)

# Draw the filtered contours on the image
cv2.drawContours(image_filtered_contours, filtered_contours, -1, (0, 255, 0), 2)

# Display the image with filtered contours drawn
cv2.imshow("Filtered Contours (After Area Filtering)", image_filtered_contours)

# Save the result as a new image
cv2.imwrite("contours_output.jpg", image_filtered_contours)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
