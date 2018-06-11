import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import config
from recognition.image_dataset import ImageDataset


def t5acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5)


def main():
    X_train, X_test, y_train, y_test = load_data()
    test_generator, train_generator = create_data_generators(X_train, X_test, y_train, y_test)

    model = create_model()

    train_model(model, train_generator, test_generator)

    print(model.evaluate_generator(test_generator, verbose=1))
    print(model.metrics_names)

    return model


def load_data():
    dataset = ImageDataset(config.OUT_PATH, config.SEED)
    X_train, X_test, y_train, y_test = dataset.split(config.TEST_SPLIT_RATIO)
    y_train = keras.utils.to_categorical(np.array(y_train), num_classes=config.CLASSES)
    y_test = keras.utils.to_categorical(np.array(y_test), num_classes=config.CLASSES)

    return X_train, X_test, y_train, y_test


def create_data_generators(X_train, X_test, y_train, y_test):
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE, shuffle=False)
    test_generator = datagen.flow(X_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)

    return test_generator, train_generator


def create_model():
    model = Sequential()
    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(config.HIDDEN_LAYER_SIZES[0][0], activation='relu'))
    model.add(Dense(config.CLASSES, activation='softmax'))

    return model


def train_model(model, train_generator, test_generator):
    sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', t5acc])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=config.TRAIN_EXAMPLES // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=test_generator,
        validation_steps=config.TEST_EXAMPLES // config.BATCH_SIZE,
        verbose=2)

    print('acc: ', history.history['acc'])
    print('loss: ', history.history['loss'])


if __name__ == "__main__":
    main()