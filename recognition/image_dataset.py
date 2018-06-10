import glob

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

import config


class ImageDataset:
    def __init__(self, path=config.OUT_PATH, seed=1):
        self._seed = seed
        self._filenames = self.__find_image_files(path)
        self.data, self.labels = self.__get_data()

        self._normalize_labels()

    def __find_image_files(self, path):
        pattern = '/**/*.{}'.format(config.IMAGE_FORMAT)

        return [name for name in glob.glob(path + pattern, recursive=True)]

    def __get_data(self):
        X, y = list(), list()

        for filename in self._filenames[:5]:
            filename = filename.replace('\\', '/')
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            X.append(image)
            y.append(int(filename.split("/")[-2]))

        return X, y

    def split(self, ratio):
        return train_test_split(self.data, self.labels, test_size=ratio, random_state=self._seed)

    def _normalize_labels(self):
        class_dict = {name: i for i, name in enumerate(list(set(label for label in self.labels)))}

        self.labels = [class_dict[label] for label in self.labels]
