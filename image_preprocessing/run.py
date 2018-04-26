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
    X_train, y_train, X_test, y_test = list(), list(), list(), list()
    label = 0
    prev = "0539"

    for i, filename in enumerate(image_files):
        print(filename, '({} / {})'.format(i, len(image_files)))

        data, label, prev = preprocess_image(filename, bbox_collection, preprocessor, lbp, label, prev)

        if i % 6 == 0:
            X_test.append(data)
            y_test.append(str(label))
        else:
            X_train.append(data)
            y_train.append(str(label))

    return X_train, y_train, X_test, y_test


def preprocess_image(filename, bbox_collection, preprocessor, lbp, idx, prev):
    image_data = ImageData(filename)
    bounding_box = bbox_collection.get(image_data.name)
    image_data.image = preprocessor.run(image_data.image, bounding_box, config.IMAGE_BORDER_STYLE)
    image_data.save()
    hist = lbp.describe(image_data.image)

    label = image_data.path.split("/")[-1]
    if prev != label:
        idx += 1
        prev = label

    return hist, idx, prev


def save_data(X_train, y_train, X_test, y_test):
    with open(os.path.join(config.RESOURCES_PATH, 'train_data'), 'w') as file:
        for features in X_train:
            file.write(','.join([str(feature) for feature in features]) + '\n')

    with open(os.path.join(config.RESOURCES_PATH, 'train_labels'), 'w') as file:
        for label in y_train:
            file.write(label + '\n')

    with open(os.path.join(config.RESOURCES_PATH, 'test_data'), 'w') as file:
        for features in X_test:
            file.write(','.join([str(feature) for feature in features]) + '\n')

    with open(os.path.join(config.RESOURCES_PATH, 'test_labels'), 'w') as file:
        for label in y_test:
            file.write(label + '\n')


def main():
    bbox_collection = BoundingBoxCollection()
    bbox_collection.load(filename=config.BOUNDING_BOXES_PATH)

    image_files = find_image_files(config.IN_PATH)
    X_train, y_train, X_test, y_test = preprocess(image_files, bbox_collection)

    save_data(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
