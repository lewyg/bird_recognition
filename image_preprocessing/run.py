import glob
import os

import config
from bounding_box import BoundingBoxCollection
from image import ImagePreprocessor, ImageData
from lbp import LocalBinaryPattern


def find_image_files(path):
    pattern = '/**/*.{}'.format(config.IMAGE_FORMAT)

    return [name for name in glob.glob(path + pattern, recursive=True)]


def preprocess(image_files, bbox_collection):
    preprocessor = ImagePreprocessor(size=config.IMAGE_SIZE)
    lbp = LocalBinaryPattern(n_points=config.LBP_POINTS_NUMBER, radius=config.LBP_RADIUS)
    X, y = list(), list()

    for i, filename in enumerate(image_files):
        print(filename, '({} / {})'.format(i, len(image_files)))

        data, label = preprocess_image(filename, bbox_collection, preprocessor, lbp)
        X.append(data)
        y.append(label)

    return X, y


def preprocess_image(filename, bbox_collection, preprocessor, lbp):
    image_data = ImageData(filename)
    bounding_box = bbox_collection.get(image_data.name)
    image_data.image = preprocessor.run(image_data.image, bounding_box, config.IMAGE_BORDER_STYLE)
    image_data.save()
    hist = lbp.describe(image_data.image)

    return hist, image_data.path.split("/")[-1]


def save_data(X, y):
    with open(os.path.join(config.RESOURCES_PATH, 'data'), 'w') as file:
        for features in X:
            file.write(','.join([str(feature) for feature in features]) + '\n')

    with open(os.path.join(config.RESOURCES_PATH, 'labels'), 'w') as file:
        for label in y:
            file.write(label + '\n')


def main():
    bbox_collection = BoundingBoxCollection()
    bbox_collection.load(filename=config.BOUNDING_BOXES_PATH)

    image_files = find_image_files(config.IN_PATH)[:100]
    X, y = preprocess(image_files, bbox_collection)

    save_data(X, y)


if __name__ == '__main__':
    main()