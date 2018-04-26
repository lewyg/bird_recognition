import keras
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, np

X_train, y_train, X_test, y_test = list(), list(), list(), list()

with open('resources/train_data') as file:
    for line in file.readlines():
        features = list(map(float, line.split(',')))
        X_train.append(features)
with open('resources/train_labels') as file:
    for line in file.readlines():
        label = int(line)
        y_train.append(label)

with open('resources/test_data') as file:
    for line in file.readlines():
        features = list(map(float, line.split(',')))
        X_test.append(features)
with open('resources/test_labels') as file:
    for line in file.readlines():
        label = int(line)
        y_test.append(label)

#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(28, 56, 112, 224), random_state=1)
#clf.fit(X_train, y_train)

# One Layer
# Layer 1: 420

# model = Sequential()
# model.add(Dense(420, activation='relu', input_dim=10))
# model.add(Dense(50, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# Two Layers
# Layer 1: 140
# Layer 2: 280

# model = Sequential()
# model.add(Dense(140, activation='relu', input_dim=10))
# model.add(Dense(280, activation='relu'))
# model.add(Dense(50, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# Three Layers
# Layer 1: 60
# Layer 2: 120
# Layer 3: 240

# model = Sequential()
# model.add(Dense(60, activation='relu', input_dim=10))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(240, activation='relu'))
# model.add(Dense(50, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# Four Layers
# Layer 1: 28
# Layer 2: 56
# Layer 3: 112
# Layer 4: 224

model = Sequential()
model.add(Dense(28, activation='relu', input_dim=10))
model.add(Dense(56, activation='relu'))
model.add(Dense(112, activation='relu'))
model.add(Dense(224, activation='relu'))
model.add(Dense(50, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Y_train = keras.utils.to_categorical(np.array(y_train), num_classes=50)
Y_test = keras.utils.to_categorical(np.array(y_test), num_classes=50)

model.fit(np.array(X_train), Y_train, epochs=20, batch_size=32)
print(model.evaluate(np.array(X_test), Y_test, batch_size=32))

#print(clf.score(X_test, y_test))
#print(clf.classes_)
