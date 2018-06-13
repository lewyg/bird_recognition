from datetime import datetime

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn import svm

import config
from recognition.image_dataset import ImageDataset


def main(bottleneck_ready=False, layers_removed=3):
    X_train, X_test, y_train, y_test = load_data()
    test_generator, train_generator = create_data_generators(X_train, X_test, y_train, y_test)

    X_test, X_train = load_bottleneck_features(bottleneck_ready, train_generator, test_generator, layers_removed)

    # while len(y_train) < len(X_train):
    #     y_train += y_train
    #
    # y_train = y_train[:len(X_train)]

    print(f"Train: {len(X_train)}, test: {len(X_test)}")

    for setting in config.SVM_SETTING:
        start = datetime.now()
        model = svm.SVC(**setting[1])
        model.fit(X_train, y_train)

        print(setting[0])
        print('  ', model.score(X_test, y_test))
        print('  ', datetime.now() - start)


def create_model(weights_path=None, layers_removed=3):
    model = Sequential()
    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)

    # Block 1
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3 - optional
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Classification block
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(config.HIDDEN_LAYER_SIZES[0][0], activation='relu'))
    model.add(Dense(config.CLASSES, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    for _ in range(layers_removed):
        model.pop()

    return model


def load_bottleneck_features(bottleneck_ready, train_generator, test_generator, layers_removed):
    if not bottleneck_ready:
        bottleneck_train, bottleneck_test = predict_bottleneck_features(train_generator, test_generator, layers_removed)

    else:
        bottleneck_train = np.load(config.SVM_BOTTLENECK_TRAIN_FEATURES_PATH + str(layers_removed))
        bottleneck_test = np.load(config.SVM_BOTTLENECK_TEST_FEATURES_PATH + str(layers_removed))

    return bottleneck_test, bottleneck_train


def predict_bottleneck_features(train_generator, test_generator, layers_removed):
    model = create_model(weights_path=config.BEST_MODEL_PATH, layers_removed=layers_removed)

    bottleneck_features_train = model.predict_generator(generator=train_generator,
                                                        max_queue_size=config.TRAIN_EXAMPLES // config.BATCH_SIZE,
                                                        verbose=1)
    np.save(file=config.SVM_BOTTLENECK_TRAIN_FEATURES_PATH + str(layers_removed), arr=bottleneck_features_train)

    bottleneck_features_test = model.predict_generator(generator=test_generator,
                                                       max_queue_size=config.TEST_EXAMPLES // config.BATCH_SIZE,
                                                       verbose=1)
    np.save(file=config.SVM_BOTTLENECK_TEST_FEATURES_PATH + str(layers_removed), arr=bottleneck_features_test)

    return bottleneck_features_train, bottleneck_features_test


def load_data():
    dataset = ImageDataset(config.OUT_PATH, config.SEED)
    X_train, X_test, y_train, y_test = dataset.split(config.TEST_SPLIT_RATIO)

    return X_train, X_test, y_train, y_test


def normal_generator():
    return ImageDataGenerator(rescale=1. / 255)


def rotate_translate_generator():
    return ImageDataGenerator(rescale=1. / 255,
                              rotation_range=8,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              fill_mode="reflect")


def create_data_generators(X_train, X_test, y_train, y_test):
    datagen_test = normal_generator()
    dategen_train = rotate_translate_generator()

    y_train = keras.utils.to_categorical(np.array(y_train), num_classes=config.CLASSES)
    y_test = keras.utils.to_categorical(np.array(y_test), num_classes=config.CLASSES)

    train_generator = datagen_test.flow(X_train, y_train, batch_size=config.BATCH_SIZE, shuffle=False)
    test_generator = datagen_test.flow(X_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)

    return test_generator, train_generator


if __name__ == "__main__":
    main(bottleneck_ready=True, layers_removed=3)
