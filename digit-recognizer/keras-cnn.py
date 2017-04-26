from keras.models import Sequential
from keras.utils import np_utils
from keras import optimizers
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense, Reshape
from keras import optimizers
from keras.layers.normalization import BatchNormalization

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('./train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels)

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale

mean = np.std(X_train)
X_train -= mean
X_train = np.expand_dims(X_train,1)

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

def get_cnn():
  model = Sequential()
  model.add(Reshape((1, 28, 28), input_shape=(1, 28 * 28)))
  model.add(Conv2D(128, 3, 3))
  # model.add(BatchNormalization())
  model.add(Activation("relu"))

  # model.add(Conv2D(128, 3, 3))
  # model.add(BatchNormalization())
  # model.add(Activation("relu"))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Dropout(0.25))

  model.add(Conv2D(16, 3, 3))
  # model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Flatten())
  # model.add(BatchNormalization())
  model.add(Dense(10, activation='softmax'))

  return model

def get_mlp():
  model = Sequential()
  model.add(Reshape((28 * 28), input_shape=(1, 28 * 28, )))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  return model

model = get_cnn()
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

print("Training...")

history = model.fit(X_train, y_train, verbose=1, nb_epoch=2, batch_size=64, validation_split=0.2)
# sgd.lr = sgd.lr * 0.5;
# history = model.fit(X_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_split=0.2)
# sgd.lr = sgd.lr * 0.5;
# history = model.fit(X_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_split=0.2)
# sgd.lr = sgd.lr * 0.5;
# history = model.fit(X_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_split=0.2)

print("Generating test predictions...")

X_test = (pd.read_csv('./test.csv').values).astype('float32')
X_test /= scale
X_test -= mean
X_test = np.expand_dims(X_test, 1)
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-cnn.csv")
