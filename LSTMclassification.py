import numpy as np

from prepare_data import read_file

files = ["data/20220418_EVLP818_converted.csv",
         "data/20220417_EVLP817_converted.csv",
         "data/20220218_EVLP803_converted.csv",
         "data/20210925_EVLP782_converted.csv",
         "data/20210620_EVLP762_converted.csv",
         "data/20210521_EVLP753_converted.csv"]

_, x_test, y_test = read_file(files[0])
x_test = x_test[1000:-1000]
y_test = y_test[1000:-1000]

_, x_train, y_train = read_file(files[1])
x_train = x_train[1000:-1000]
y_train = y_train[1000:-1000]

_, x_train2, y_train2 = read_file(files[2])
x_train = np.concatenate((x_train, x_train2[1000:-1000]))
y_train = np.concatenate((y_train, y_train2[1000:-1000]))

_, x_train2, y_train2 = read_file(files[3])
x_train = np.concatenate((x_train, x_train2[1000:-1000]))
y_train = np.concatenate((y_train, y_train2[1000:-1000]))

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# use LSTM to segment the data where X is an 1-d time series and y is an 1-d time series of binary labels
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 1))
x_test = scaler.transform(x_test.reshape(-1, 1))

# reshape the data
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# build the model
model = Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')])
model.summary()

# evaluate the model
y_pred = model.predict_classes(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot the training loss and accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()


