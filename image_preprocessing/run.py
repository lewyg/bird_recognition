import glob

import config
from bounding_box import BoundingBoxCollection
from image import ImagePreprocessor, Image


def find_image_files():
    pattern = '/**/*.{}'.format(config.IMAGE_FORMAT)

    return [name for name in glob.glob(config.IN_PATH + pattern, recursive=True)]


def preprocess(image_files, bbox_collection, preprocessor):
    for i, filename in enumerate(image_files):
        print(filename, '({} / {})'.format(i, len(image_files)))

        image_data = Image(filename)
        bounding_box = bbox_collection.get(image_data.name)

        image_data.image = preprocessor.run(image_data.image, bounding_box, config.IMAGE_BORDER_STYLE)

        image_data.save()


def main():
    bbox_collection = BoundingBoxCollection()
    bbox_collection.load(filename=config.BOUNDING_BOXES_PATH)

    image_files = find_image_files()
    preprocessor = ImagePreprocessor(size=config.IMAGE_SIZE)

    preprocess(image_files, bbox_collection, preprocessor)


if __name__ == '__main__':
    main()
