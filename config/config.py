import numpy as np

image2contours_kwargs = dict(
    epsilon=2.5, lowercut=100, area_lowercut=1000, gaussian_window=np.array([7, 7])
)
