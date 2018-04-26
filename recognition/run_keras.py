import keras
from keras.layers import Dense, np
from keras.models import Sequential

import config
from recognition.dataset import Dataset


def main(layers=4):
    dataset = Dataset(config.DATA_PATH, config.LABELS_PATH)
    X_train, X_test, y_train, y_test = dataset.split(ratio=0.7)

    input_layer_size = len(X_train[0])
    output_layer_size = len(dataset.classes())
    hidden_layer_sizes = config.HIDDEN_LAYER_SIZES[layers-1]

    clf = Sequential()

    clf.add(Dense(hidden_layer_sizes[0], activation='relu', input_dim=input_layer_size))

    for layer_size in hidden_layer_sizes[1:]:
        clf.add(Dense(layer_size, activation='relu'))

    clf.add(Dense(output_layer_size, activation='softmax'))

    clf.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    y_train = keras.utils.to_categorical(np.array(y_train), num_classes=output_layer_size)
    y_test = keras.utils.to_categorical(np.array(y_test), num_classes=output_layer_size)

    clf.fit(np.array(X_train), y_train, epochs=20, batch_size=32)

    print(clf.evaluate(np.array(X_test), y_test, batch_size=32))


if __name__ == '__main__':
    main()