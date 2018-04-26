import cv2

import numpy as np
from skimage.feature import local_binary_pattern


class LocalBinaryPattern:
    def __init__(self, n_points, radius, method):
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lbp = local_binary_pattern(image, self.n_points, self.radius, self.method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        return hist
