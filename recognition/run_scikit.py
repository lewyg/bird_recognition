from sklearn.neural_network import MLPClassifier

import config
from recognition.dataset import Dataset


def main():
    dataset = Dataset(config.DATA_PATH, config.LABELS_PATH)
    X_train, X_test, y_train, y_test = dataset.split(ratio=0.7)

    layer_sizes = (28, 56, 112, 224)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layer_sizes, random_state=1)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
