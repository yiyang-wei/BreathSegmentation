from prepare_data import read_file, concat_files

import numpy as np

files = ["data/20220418_EVLP818_converted.csv"
         "data/20220417_EVLP817_converted.csv",
         "data/20220218_EVLP803_converted.csv",
         "data/20210925_EVLP782_converted.csv",
         "data/20210620_EVLP762_converted.csv",
         "data/20210521_EVLP753_converted.csv"]


WINDOW_SIZE = 601

X_train, y_train = concat_files(files[1:], window_size=WINDOW_SIZE)
X_test, y_test = concat_files(files[:1], window_size=WINDOW_SIZE)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# use CNN to classify
import tensorflow as tf
from tensorflow.keras import layers, models

# print device name
print(tf.config.list_physical_devices('GPU'))

# TODO: try dilation rate
model = models.Sequential()
model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(WINDOW_SIZE, 1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3,
                    validation_data=(X_test, y_test))

# plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(test_acc)

