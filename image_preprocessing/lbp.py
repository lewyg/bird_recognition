import cv2
import numpy as np
from skimage import feature


class LocalBinaryPattern:
    def __init__(self, n_points, radius):
        self.n_points = n_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
