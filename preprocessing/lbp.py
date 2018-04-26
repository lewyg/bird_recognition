import cv2

import numpy as np
from skimage.feature import local_binary_pattern


class LocalBinaryPattern:
    def __init__(self, n_points, radius, method, part_size):
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.part_size = part_size

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = list()

        n_bins = self.n_points + 2

        for y in range(0, image.shape[0], self.part_size):
            for x in range(0, image.shape[1], self.part_size):
                window = image[y:y + self.part_size, x:x + self.part_size]

                lbp = local_binary_pattern(window, self.n_points, self.radius, self.method)
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                histogram.extend(hist)

        return histogram
