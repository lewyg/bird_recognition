import cv2
import config

import numpy as np
from skimage.feature import local_binary_pattern


class LocalBinaryPattern:
    def __init__(self, n_points, radius, method):
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = list()

        for r in range(0, image.shape[0] + 1 - config.PART_SIZE, config.PART_SIZE):
            for c in range(0, image.shape[1] + 1 - config.PART_SIZE, config.PART_SIZE):
                window = image[r:r + config.PART_SIZE, c:c + config.PART_SIZE]
                lbp = local_binary_pattern(window, self.n_points, self.radius, self.method)
                n_bins = int(lbp.max() + 1)
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                histogram.extend(hist)

        return np.array(histogram)
