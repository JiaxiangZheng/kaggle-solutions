import keras as keras
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train.species = train.species.astype("category")

# from the index to categorical string
mapping = zip(train.species.cat.codes, train.species)
mapping = dict(mapping)
columns = [mapping[index] for index in range(99)]

cat_columns = train.select_dtypes(['category']).columns
train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)

train_Y = train.species.values
train_Y = to_categorical(train_Y)

# print train.columns
train.drop(['id', 'species'], inplace=True, axis=1)

train_X = train.values.astype('float64')
N, D = train_X.shape
print train_X.shape, train_Y.shape
# pd.DataFrame(data=train_X).to_csv('train_x.csv', index=False)
# pd.DataFrame(data=train_Y).to_csv('train_y.csv', index=False)

def get_simple_nn():
  model = Sequential()
  model.add(Dense(1024,input_dim=192))
  model.add(Dropout(0.2))
  model.add(Activation('relu'))
  model.add(Dense(512))
  model.add(Dropout(0.3))
  model.add(Activation('relu'))
  model.add(Dense(99))
  model.add(Activation('softmax'))
  return model

def get_mlp():
  model = Sequential()
  model.add(Dense(512, input_shape=(D, )))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(99))
  model.add(Activation('softmax'))
  return model

def get_linear():
  model = Sequential()
  model.add(Dense(99, input_shape=(D, ), activation='softmax'))
  return model

# model = get_mlp()
# model = get_linear();
model = get_simple_nn()
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = model.fit(train_X, train_Y, validation_split=0.1, nb_epoch=100, batch_size=64)

ids = test.id
test.drop(['id'], axis=1, inplace=True)
test_X = test.values.astype('float64')
test_Y = model.predict(test_X)

# print ids.values
# print test_Y    # N_test * 99

df = pd.DataFrame(data=test_Y, columns=columns)
df.insert(0, 'id', ids)
df.to_csv('submission.csv', index=False)

