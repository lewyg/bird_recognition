import cv2
import glob

import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

import config
from preprocessing.image import ImageData
from recognition.dataset import Dataset


def VGG_16(weights_path=None):
    vgg_16 = Sequential()
    vgg_16.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    vgg_16.add(Conv2D(64, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(64, (3, 3), activation='relu'))
    vgg_16.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(128, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(128, (3, 3), activation='relu'))
    vgg_16.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(256, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(256, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(256, (3, 3), activation='relu'))
    vgg_16.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(512, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(512, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(512, (3, 3), activation='relu'))
    vgg_16.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(512, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(512, (3, 3), activation='relu'))
    vgg_16.add(ZeroPadding2D((1, 1)))
    vgg_16.add(Conv2D(512, (3, 3), activation='relu'))
    vgg_16.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    vgg_16.add(Flatten())
    vgg_16.add(Dense(4096, activation='relu'))
    vgg_16.add(Dropout(0.5))
    vgg_16.add(Dense(4096, activation='relu'))
    vgg_16.add(Dropout(0.5))
    vgg_16.add(Dense(1000, activation='softmax'))

    if weights_path:
        vgg_16.load_weights(weights_path, by_name=True)

    return vgg_16


def find_image_files(path):
    pattern = '/**/*.{}'.format(config.IMAGE_FORMAT)

    return [name for name in glob.glob(path + pattern, recursive=True)]


def get_data(filenames):
    X, y = list(), list()
    for filename in filenames:
        filename = filename.replace('\\', '/')
        X.append(cv2.imread(filename))
        y.append(int(filename.split("/")[-2]))

    return np.array(X), np.array(y)


def main():
    X, y = get_data(find_image_files(config.OUT_PATH))

    dataset = Dataset(config.DATA_PATH, config.LABELS_PATH)
    X_train, X_test, y_train, y_test = dataset.split(ratio=0.7)
    # Test pretrained model
    model = VGG_16(config.VGG16_WEIGHTS_PATH)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # print(model.evaluate(np.array(X_test), y_test, batch_size=32))
    out = model.predict(np.array(X_test))
    print(np.argmax(out))


if __name__ == "__main__":
    main()
