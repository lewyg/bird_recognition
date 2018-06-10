import glob

import cv2
import keras
import numpy
import numpy as np
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

import config
from recognition.image_dataset import ImageDataset


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


def save_bottlebeck_features(train_generator, test_generator):
    model = VGG16(include_top=False, weights='imagenet')

    # generator = datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    #bottleneck_features_train = model.predict(X_train)
    bottleneck_features_train = model.predict_generator(
        train_generator, config.TRAIN_EXAMPLES // config.BATCH_SIZE, verbose=1)
    np.save(config.BOTTLENECK_TRAIN_FEATURES_PATH, bottleneck_features_train)

    # generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    #bottleneck_features_test = model.predict(X_test)
    bottleneck_features_test = model.predict_generator(
        test_generator, config.TEST_EXAMPLES // config.BATCH_SIZE, verbose=1)
    np.save(config.BOTTLENECK_TEST_FEATURES_PATH, bottleneck_features_test)


def train_top_model(train_labels, test_labels):
    train_data = np.load(config.BOTTLENECK_TRAIN_FEATURES_PATH)
    test_data = np.load(config.BOTTLENECK_TEST_FEATURES_PATH)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(config.HIDDEN_LAYER_SIZES[0][0], activation='relu'))
    model.add(Dense(config.CLASSES, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE,
              validation_data=(test_data, test_labels),
              verbose=1)

    model.save_weights(config.TOP_MODEL_WEIGHTS_PATH)


def main():
    dataset = ImageDataset(config.OUT_PATH, config.SEED)
    X_train, X_test, y_train, y_test = dataset.split(1 - config.TEST_SPLIT_RATIO)

    y_train = keras.utils.to_categorical(np.array(y_train), num_classes=config.CLASSES)
    y_test = keras.utils.to_categorical(np.array(y_test), num_classes=config.CLASSES)

    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow(
        X_train,
        y_train,
        batch_size=config.BATCH_SIZE,
        shuffle=False)

    test_generator = datagen.flow(
        X_test,
        y_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False)

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

    # save_bottlebeck_features(train_generator, test_generator)
    train_top_model(y_train, y_test)


if __name__ == "__main__":
    main()
