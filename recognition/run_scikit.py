from sklearn.neural_network import MLPClassifier

import config
from recognition.dataset import Dataset


def main(layers=1):
    dataset = Dataset(config.DATA_PATH, config.LABELS_PATH)
    X_train, X_test, y_train, y_test = dataset.split(ratio=0.7)

    hidden_layer_sizes = config.HIDDEN_LAYER_SIZES[layers - 1]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes,
                        random_state=1, max_iter=config.PERCEPTRON_MAX_EPOCHS)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
