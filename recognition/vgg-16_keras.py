import glob

import cv2
import keras
import numpy
import numpy as np
from keras import layers, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split

import config


def find_image_files(path):
    pattern = '/**/*.{}'.format(config.IMAGE_FORMAT)

    return [name for name in glob.glob(path + pattern, recursive=True)]


def get_data(filenames):
    X, y = list(), list()
    for filename in filenames:
        filename = filename.replace('\\', '/')
        image = cv2.imread(filename)
        X.append(image)
        y.append(int(filename.split("/")[-2]))

    return np.array(X), np.array(y)


def save_bottlebeck_features(X_train, X_test):
    # datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    # generator = datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    bottleneck_features_train = model.predict(X_train)
    # with open(config.BOTTLENECK_TRAIN_FEATURES_PATH, 'w') as file:
    #     file.write(str(bottleneck_features_train))

    np.save(config.BOTTLENECK_TRAIN_FEATURES_PATH, bottleneck_features_train)

    # generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    bottleneck_features_test = model.predict(X_test)
    # with open(config.BOTTLENECK_TEST_FEATURES_PATH, 'w') as file:
    #     file.write(str(bottleneck_features_validation))

    np.save(config.BOTTLENECK_TEST_FEATURES_PATH, bottleneck_features_test)


def train_top_model(train_labels, test_labels):
    train_data = np.load(config.BOTTLENECK_TRAIN_FEATURES_PATH)
    test_data = np.load(config.BOTTLENECK_TEST_FEATURES_PATH)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(420, activation='relu'))
    model.add(Dense(50, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=50,
              batch_size=32,
              validation_data=(test_data, test_labels))

    model.save_weights(config.TOP_MODEL_WEIGHTS_PATH)


def main():

    X, y = get_data(find_image_files(config.OUT_PATH))
    y = keras.utils.to_categorical(y, num_classes=50)
    seed = 7
    numpy.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    # y_train = keras.utils.to_categorical(np.array(y_train), num_classes=50)
    # y_test = keras.utils.to_categorical(np.array(y_test), num_classes=50)

    # model = VGG16(include_top=False, weights="imagenet", classes=1000,
    #               input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    # print(model.summary())

    #
    # # Add top model
    # top_model = Sequential()
    # top_model.add(layers.Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(layers.Dense(420, activation='relu'))
    # top_model.add(layers.Dense(50, activation='softmax'))
    # top_model.load_weights(config.TOP_MODEL_WEIGHTS_PATH)

    # model.add(top_model)

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['acc'])

    #model.fit(X_train, y_train, batch_size=100, epochs=30)
    #print(model.evaluate(X_test, y_test, batch_size=32))

    # Save the model
   # model.save('vgg16_model')

    save_bottlebeck_features(X_train[1:3], X_test[1:3])
    train_top_model(y_train, y_test)


if __name__ == "__main__":
    main()
