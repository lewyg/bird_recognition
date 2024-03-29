import os

import keras
import keras.metrics as keras_metrics
import matplotlib.pyplot as plt
from keras.layers import Dense, np
from keras.models import Sequential
from scipy import interp
from sklearn.metrics import roc_curve, auc

import config
from recognition.dataset import Dataset


def figure_path(name):
    return os.path.join(config.PLOT_PATH, name)


def t5er(y_true, y_pred):
    return keras_metrics.top_k_categorical_accuracy(y_true, y_pred, 5)


def main(layers=4):
    dataset = Dataset(config.DATA_PATH, config.LABELS_PATH)
    X_train, X_test, y_train, y_test = dataset.split(ratio=0.7)

    input_layer_size = len(X_train[0])
    output_layer_size = len(dataset.classes())
    hidden_layer_sizes = config.HIDDEN_LAYER_SIZES[layers - 1]

    clf = Sequential()

    clf.add(Dense(hidden_layer_sizes[0], activation='relu', input_dim=input_layer_size))

    for layer_size in hidden_layer_sizes[1:]:
        clf.add(Dense(layer_size, activation='relu'))

    clf.add(Dense(output_layer_size, activation='softmax'))

    clf.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', t5er])

    y_train = keras.utils.to_categorical(np.array(y_train), num_classes=output_layer_size)
    y_test = keras.utils.to_categorical(np.array(y_test), num_classes=output_layer_size)

    history = clf.fit(np.array(X_train), y_train, epochs=config.PERCEPTRON_MAX_EPOCHS,
                      batch_size=32, validation_data=(np.array(X_test), y_test))

    print(clf.evaluate(np.array(X_test), y_test, batch_size=32))

    plot_training_accuracy(history, layers)
    plot_validation_accuracy(history, layers)

    create_roc(clf, X_test, y_test, layers, output_layer_size)


def create_roc(clf, X_test, y_test, layers, output_layer_size):
    y_score = clf.predict(np.array(X_test))
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    n_classes = output_layer_size
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to bird_classification')
    plt.legend(loc="lower right")
    plt.axes().set_aspect('equal')
    plt.grid(True)

    name = 'plot_roc{}_lbp_radius{}.png'.format(layers, config.LBP_RADIUS)

    plt.savefig(figure_path(name))
    plt.clf()


def plot_training_accuracy(history, layers):
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.plot(history.history['acc'], label='top-1')
    plt.plot(history.history['t5er'], label='top-5')
    plt.legend(loc="lower right")

    name = 'plot_acc_test{}_lbp_radius{}.png'.format(layers, config.LBP_RADIUS)

    plt.savefig(figure_path(name))
    plt.clf()


def plot_validation_accuracy(history, layers):
    plt.xlabel('Numer epoki')
    plt.ylabel('Accuracy')
    plt.plot(history.history['val_acc'], label='top-1')
    plt.plot(history.history['val_t5er'], label='top-5')
    plt.legend(loc="lower right")

    name = 'plot_acc_val{}_lbp_radius{}.png'.format(layers, config.LBP_RADIUS)

    plt.savefig(figure_path(name))
    plt.clf()


if __name__ == '__main__':
    main(1)
    # main(2)
    # main(3)
    # main(4)
