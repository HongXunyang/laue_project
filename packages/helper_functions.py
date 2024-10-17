import numpy as np
import cv2


def _rotate(center, point, phi_offset):
    """Rotate a point around the center, compenate the phi_offset

    Args:
    - center: center of the rotation
    - point: point to rotate
    - phi_offset: the angle to rotate, in degree, counter-clockwise
    """
    x, y = point  # -1, 0
    cx, cy = center  # 0, 0
    # target 0, 1
    # phi_offset = 90
    phi_to_rotate = -phi_offset * np.pi / 180
    x_new = (x - cx) * np.cos(phi_to_rotate) - (y - cy) * np.sin(phi_to_rotate) + cx
    y_new = (x - cx) * np.sin(phi_to_rotate) + (y - cy) * np.cos(phi_to_rotate) + cy
    return x_new, y_new


def _contour2centroid(contour):
    M = cv2.moments(contour)
    # Ensure the area is not zero before calculating centroid
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        print(f"Contour has zero area, returning None.")
        return None


def _center_of_mass(stripes_vectors):
    return np.mean(stripes_vectors, axis=0)


def _hull2centroid(hull):
    M = cv2.moments(hull)
    # Ensure the area is not zero before calculating centroid
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        print(f"Hull has zero area, returning None.")
        return None


def _remove_background_contour(contours, hulls):
    """
    Remove the background contour from the list

    Mechanism:
    if the largest area of contour is `backgroun_sample_size_ratio` times larger than the second largest, it is the background
    """
    backgroun_sample_size_ratio = 5
    areas = [cv2.contourArea(hull) for hull in hulls]
    if len(areas) > 1:
        max_area = max(areas)
        max_index = areas.index(max_area)
        areas.pop(max_index)
        second_max_area = max(areas)
        if max_area > backgroun_sample_size_ratio * second_max_area:
            contours.pop(max_index)
            hulls.pop(max_index)
    return contours, hulls


def distance(a, b):
    return np.linalg.norm(a - b)
