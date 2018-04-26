import csv
from itertools import groupby, chain
from operator import itemgetter

import numpy as np


class Dataset:
    def __init__(self, data_file, labels_file):
        self.data = self._read_file(data_file)
        self.labels = self._read_file(labels_file)

        for data_row, label in zip(self.data, self.labels):
            data_row.append(label[0])

        class_dict = self.__get_class_dict()
        self._apply_data_types(self.data, float, class_dict.get)

    def split(self, ratio):
        training_set = list()
        test_set = list()

        grouped_data = groupby(sorted(self.data, key=itemgetter(-1)), itemgetter(-1))

        for _, key_data in grouped_data:
            permuted_data = np.random.permutation(list(key_data))
            training_set_size = int(ratio * len(permuted_data))
            training_set.extend(permuted_data[:training_set_size])
            test_set.extend(permuted_data[training_set_size:])

        return (self._columns(training_set, slice(-1)), self._columns(test_set, slice(-1)),
                self._columns(training_set, -1), self._columns(test_set, -1))

    def split_folds(self, n):
        folds = [list() for _ in range(n)]
        grouped_data = groupby(sorted(self.data, key=itemgetter(-1)), itemgetter(-1))

        for _, key_data in grouped_data:
            permuted_data = np.random.permutation(list(key_data))
            fold_size = int(len(permuted_data) / n)
            for i, fold in enumerate(folds):
                fold.extend(permuted_data[i * fold_size:(i + 1) * fold_size])

        return self._folds_iterator(folds)

    def _folds_iterator(self, folds):
        for i, test_set in enumerate(folds):
            training_set = list(chain(*folds[:i], *folds[i + 1:]))
            test_set = list(test_set)

            yield (self._columns(training_set, slice(-1)), self._columns(test_set, slice(-1)),
                   self._columns(training_set, -1), self._columns(test_set, -1))

    def classes(self):
        return list(set(sample[-1] for sample in self.data))

    def __get_class_dict(self):
        classes = self.classes()

        return {name: i for i, name in enumerate(classes)}

    @staticmethod
    def _columns(dataset, columns):
        return [row_data[columns] for row_data in dataset]

    @staticmethod
    def _read_file(filename):
        with open(filename, 'r') as file:
            return list(csv.reader(file))

    @staticmethod
    def _apply_data_types(data, feature_type=float, class_type=int):
        for sample in data:
            sample[:-1] = list(map(feature_type, sample[:-1]))
            sample[-1] = class_type(sample[-1])
