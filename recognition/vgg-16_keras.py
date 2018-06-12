import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import config
from recognition.image_dataset import ImageDataset


def t5acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5)


def main(bottleneck_ready=False, top_model_ready=False, ground_truth_ready=False):
    X_train, X_test, y_train, y_test = load_data()
    test_generator, train_generator = create_data_generators(X_train, X_test, y_train, y_test)

    bottleneck_test, bottleneck_train = load_bottleneck_features(bottleneck_ready, train_generator, test_generator)
    model = build_model(top_model_ready, bottleneck_train, bottleneck_test, y_train, y_test)

    fine_tune_model(ground_truth_ready, model, train_generator, test_generator)

    print(model.evaluate_generator(test_generator))
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


def load_bottleneck_features(bottleneck_ready, train_generator, test_generator):
    if not bottleneck_ready:
        bottleneck_train, bottleneck_test = predict_bottleneck_features(train_generator, test_generator)

    else:
        bottleneck_train = np.load(config.BOTTLENECK_TRAIN_FEATURES_PATH)
        bottleneck_test = np.load(config.BOTTLENECK_TEST_FEATURES_PATH)

    return bottleneck_test, bottleneck_train


def predict_bottleneck_features(train_generator, test_generator):
    model = VGG16(include_top=False, weights='imagenet')

    bottleneck_features_train = model.predict_generator(generator=train_generator,
                                                        max_queue_size=config.TRAIN_EXAMPLES // config.BATCH_SIZE,
                                                        verbose=1)
    np.save(file=config.BOTTLENECK_TRAIN_FEATURES_PATH, arr=bottleneck_features_train)

    bottleneck_features_test = model.predict_generator(generator=test_generator,
                                                       max_queue_size=config.TEST_EXAMPLES // config.BATCH_SIZE,
                                                       verbose=1)
    np.save(file=config.BOTTLENECK_TEST_FEATURES_PATH, arr=bottleneck_features_test)

    return bottleneck_features_train, bottleneck_features_test


def build_model(top_model_ready, bottleneck_train, bottleneck_test, y_train, y_test):
    model = VGG16(include_top=False, weights="imagenet", input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    top_model = create_top_model(model.output_shape[1:])

    sgd = SGD(lr=1e-4, momentum=0.9, nesterov=True)
    top_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', t5acc])

    if top_model_ready:
        top_model.load_weights(config.TOP_MODEL_WEIGHTS_PATH)

    else:
        train_top_model(top_model, bottleneck_train, y_train, bottleneck_test, y_test)

    print(top_model.evaluate(bottleneck_test, y_test, batch_size=32))

    model = Model(inputs=model.input, outputs=top_model(model.output))

    return model


def create_top_model(input_shape):
    model = Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(Dense(config.HIDDEN_LAYER_SIZES[0][0], activation='relu'))
    model.add(Dense(config.CLASSES, activation='softmax'))

    return model


def train_top_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train,
                        epochs=3,
                        batch_size=config.BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        verbose=1)

    plot_history(history)

    model.save_weights(config.TOP_MODEL_WEIGHTS_PATH)


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

    name = 'tommodel_history.png'.format(layers, config.LBP_RADIUS)

    plt.savefig(figure_path(name))
    plt.clf()


def fine_tune_model(ground_truth_ready, model, train_generator, test_generator):
    for layer in model.layers[:15]:
        layer.trainable = False

    sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', t5acc])

    if ground_truth_ready:
        model.load_weights(config.GROUND_TRUTH_PATH)

    else:
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=config.TRAIN_EXAMPLES // config.BATCH_SIZE,
            epochs=config.VGG_EPOCHS,
            validation_data=test_generator,
            validation_steps=config.TEST_EXAMPLES // config.BATCH_SIZE,
            verbose=2)

        print('acc: ', history.history['acc'])
        print('loss: ', history.history['loss'])

        model.save(config.GROUND_TRUTH_PATH)


if __name__ == "__main__":
    main(bottleneck_ready=True, top_model_ready=False, ground_truth_ready=False)
