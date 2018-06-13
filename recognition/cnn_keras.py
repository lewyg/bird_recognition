import os
from random import random

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.legacy import layers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import config
from recognition.image_dataset import ImageDataset


def t5acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5)


def figure_path(name):
    return os.path.join(config.PLOT_PATH, name)


def plot_history(history):
    plt.xlabel('Numer epoki')
    plt.ylabel('Accuracy')
    plt.plot(history.history['acc'], label='train_top-1')
    plt.plot(history.history['t5acc'], label='train_top-5')
    plt.plot(history.history['val_acc'], label='val_top-1')
    plt.plot(history.history['val_t5acc'], label='val_top-5')
    plt.legend(loc="lower right")

    name = 'cnn_hide_history.png'.format(layers, config.LBP_RADIUS)

    plt.savefig(figure_path(name))
    plt.clf()


def main(datagen):
    X_train, X_test, y_train, y_test = load_data()
    test_generator, train_generator = create_data_generators(X_train, X_test, y_train, y_test, datagen)

    model = create_model()

    train_model(model, train_generator, test_generator)

    print(model.evaluate_generator(test_generator))
    print(model.metrics_names)

    return model


def hide_part(img):
    if random() > 0.6:
        size_a = int(random() * 80) + 20
        size_b = int(random() * 80) + 20
        a = int(random() * (223 - size_a))
        b = int(random() * (223 - size_b))
        img[a:a + size_a, b:b + size_b] = np.zeros((size_a, size_b, 3))


def add_noise(img):
    if np.random.random() > 0.3:
        noise_shape = img.shape
        noise = 10 * np.random.randn(*noise_shape)
        img[:] = img + noise


def load_data():
    dataset = ImageDataset(config.OUT_PATH, config.SEED)
    X_train, X_test, y_train, y_test = dataset.split(config.TEST_SPLIT_RATIO)

    # # apply noise
    # for img in X_train:
    #     add_noise(img)

    # apply hide_part
    for img in X_train:
        hide_part(img)

    y_train = keras.utils.to_categorical(np.array(y_train), num_classes=config.CLASSES)
    y_test = keras.utils.to_categorical(np.array(y_test), num_classes=config.CLASSES)

    return X_train, X_test, y_train, y_test


def normal_generator():
    return ImageDataGenerator(rescale=1. / 255)


def rotate_generator():
    return ImageDataGenerator(rescale=1. / 255,
                              rotation_range=8,
                              fill_mode="reflect")


def translate_generator():
    return ImageDataGenerator(rescale=1. / 255,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              fill_mode="reflect")


def rotate_translate_generator():
    return ImageDataGenerator(rescale=1. / 255,
                              rotation_range=8,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              fill_mode="reflect")


def create_data_generators(X_train, X_test, y_train, y_test, datagen):
    datagen_test = normal_generator()
    train_generator = datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE, shuffle=False)
    test_generator = datagen_test.flow(X_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)

    return test_generator, train_generator


def create_model():
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

    return model


def train_model(model, train_generator, test_generator):
    learning_rate = 0.005
    epochs = 200
    decay_rate = learning_rate / epochs
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', t5acc])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=config.TRAIN_EXAMPLES // config.BATCH_SIZE,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=config.TEST_EXAMPLES // config.BATCH_SIZE,
        verbose=2)

    model.save(config.HIDE)

    print('acc: ', history.history['acc'])
    print('loss: ', history.history['loss'])

    plot_history(history)


if __name__ == "__main__":
    datagen = rotate_translate_generator()

    main(datagen)
