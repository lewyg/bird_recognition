import os

import cv2

import config


def mkdir(path):
    try:
        os.makedirs(path)

    except OSError:
        pass


class Image:
    def __init__(self, filename):
        self.filename = filename.replace('\\', '/')

        self.image = self.__load()
        self.name = self.__get_name_from_path()
        self.path = self.__get_relative_path()

    def save(self):
        path = os.path.join(config.OUT_PATH, self.path)
        filename = os.path.join(path, '{}.{}'.format(self.name, config.IMAGE_FORMAT)).replace('\\', '/')

        mkdir(path)
        cv2.imwrite(filename, self.image)

    def __load(self):
        return cv2.imread(self.filename)

    def __get_name_from_path(self):
        return self.filename.split('/')[-1].split('.')[0]

    def __get_relative_path(self):
        return '/'.join(self.filename.replace(config.IN_PATH, '').split('/')[1:-1])


class ImagePreprocessor:
    def __init__(self, size):
        self.size = size

    def run(self, image, bounding_box, border_style):
        scale_ratio = min(self.size / bounding_box.width, self.size / bounding_box.height)
        x, y = self.__get_fixed_position(bounding_box)

        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)
        x = int(x * scale_ratio) + self.size
        y = int(y * scale_ratio) + self.size

        image = cv2.copyMakeBorder(image, self.size, self.size, self.size, self.size, border_style)

        return image[y:y + self.size, x:x + self.size]

    def __get_fixed_position(self, bounding_box):
        diff = abs(bounding_box.height - bounding_box.width)
        if bounding_box.height > bounding_box.width:
            return int(bounding_box.x - diff / 2), bounding_box.y

        else:
            return bounding_box.x, int(bounding_box.y - diff / 2)
