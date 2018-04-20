import os

import cv2

import config


class Image:
    def __init__(self, filename):
        self.filename = filename.replace('\\', '/')

        self.image = self.__load()
        self.name = self.__get_name_from_path()
        self.path = self.__get_relative_path()

    def save(self):
        filename = os.path.join(
            config.OUT_PATH,
            self.path,
            '{}.{}'.format(self.name, config.IMAGE_FORMAT)
        ).replace('\\', '/')

        cv2.imwrite(filename, self.image)

    def __load(self):
        return cv2.imread(self.filename)

    def __get_name_from_path(self):
        return self.filename.split('/')[-1].split('.')[0]

    def __get_relative_path(self):
        return '/'.join(self.filename.replace(config.IN_PATH, '').split('/')[1:-1])


class ImagePreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def run(self, image, bounding_box):
        scale_ratio = self.__get_scale_ratio(bounding_box)
        x, y = self.__get_square_position(bounding_box)
        x = int(x * scale_ratio)
        y = int(y * scale_ratio)

        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)
        image = image[y:y + self.height, x:x + self.width]

        return image

    def __get_scale_ratio(self, bounding_box):
        return min(self.width / bounding_box.width, self.height / bounding_box.height)

    def __get_square_position(self, bounding_box):
        diff = abs(bounding_box.height - bounding_box.width)

        if bounding_box.height > bounding_box.width:
            return bounding_box.y, int(bounding_box.x - diff / 2)

        else:
            return bounding_box.x, int(bounding_box.y - diff / 2)

