#!/usr/bin/env python
import argparse
import cPickle
import glob
import gzip
import logging
import numpy
import PIL.Image
import sys

numpy.random.seed(51244)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD

import clean

from logger import logger

def load_training(limit=None, neighbors=2):
    xs = []
    ys = []

    for path in glob.glob('./data/train/*.png')[:limit]:
        patches, _ = clean.x_from_image(path, neighbors)
        solutions = clean.y_from_image(path, neighbors)

        xs.extend(patches)
        ys.extend(solutions)

    return xs, ys

# will clip the raw images to patches with size clip_size * clip_size
def load_clip_training(limit, clip_size):
    xs = []
    ys = []

    for path in glob.glob('./data/train/*.png')[:limit]:
        patches = clean.clip_from_image(path, (clip_size, clip_size))
        solutions = clean.clip_from_image(path, (clip_size, clip_size))

        xs.extend(patches)
        ys.extend(solutions)
        print path, patches.shape, solutions.shape

    return xs, ys

def build_cnn_model(input_shape):
  model = Sequential()
  model.add(Reshape((1, ) + input_shape, input_shape=input_shape))
  model.add(Conv2D(128, 3, 3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(16, 3, 3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(10, activation='relu'))
  sgd = SGD(lr=0.01, momentum=0.9) # , nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd)

  return model

# we can try to clip the image and reduce the input image size, then generate multiple training cases
# use cleaned clips as the target
def build_model(input_size):
    model = Sequential()
    model.add(Dense(512, input_shape=(input_size,), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu'))

    sgd = SGD(lr=0.01, momentum=0.9) # , nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

def split_training(xs, ys):
    # randomly shuffle the x & y
    joined = zip(xs, ys)
    numpy.random.shuffle(joined)
    xs = [x for x, _ in joined]
    ys = [y for _, y in joined]

    # 8:1:1
    train_count = int(len(xs)*8/10.0)
    valid_count = int(len(xs)/10.0)
    test_count = valid_count

    res = (xs[:train_count], ys[:train_count], \
        xs[train_count:train_count+valid_count], ys[train_count:train_count+valid_count], \
        ys[train_count+valid_count:train_count+valid_count+test_count], ys[train_count+valid_count:train_count+valid_count+test_count])

    # why use numpy array to convert the data???
    # answer is that xs is a 2D array, so use numpy.array to convert to numpy object
    return [numpy.array(r) for r in res]

def train(limit, neighbors, epochs, batch_size):
    logger.info('start loading data %d %d' % (limit, neighbors))
    xs, ys = load_training(limit, neighbors)
    # xs, ys = load_clip_training(limit, 28)
    logger.info('finish loading data, spliting the data')

    train_x, train_y, valid_x, valid_y, test_x, test_y = split_training(xs, ys)
    print train_x.shape, train_y.shape, valid_x.shape, valid_y.shape

    # model = build_cnn_model(xs[0].shape)
    model = build_model(len(train_x[0]))
    logger.info('finish building the model')
    model.fit(train_x, train_y,
              nb_epoch=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(valid_x, valid_y))
    return model

def save_model(model, path):
    with open(path, 'w') as f:
        cPickle.dump(model, f)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', help='Set verbosity.')
    parser.add_argument('-l', '--limit', help='Number of training images to load.', type=int, default=5)
    parser.add_argument('-n', '--neighbors', help='Number of neighbors to use for network.', type=int, default=5)
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Number of epochs to run for.')
    parser.add_argument('-b', '--batch-size', default=10, type=int, help='The size of each minibatch.')
    parser.add_argument('path', default='./model.h5', help='Where to save the model.')
    args = parser.parse_args()

    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)

    model = train(limit=args.limit, neighbors=args.neighbors,
                  epochs=args.epochs, batch_size=args.batch_size)
    out = (model, {'neighbors': args.neighbors})
    save_model(out, args.path)

    return 0

if __name__ == '__main__':
    sys.exit(main())
