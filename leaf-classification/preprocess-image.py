'''This file is used to preprocess the raw images with:
1. read raw pixels and normalize it (resize)
2. combine with the original width / height ratio
'''

from random import shuffle
import os
import pickle
from PIL import Image
import numpy as np
import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

ids = train.id
species = train.species

# make dir and move all the images to corresponding directory
# os.mkdir('./train')
train = train.values[:, :2]
N, D = train.shape

# TODO: directy read pixel datas and dump as a python object
# image width / height ratio as a feature
# image boundary and edges as a feature
# for each epoch, run model for each batch_size
def run_epoch(dir, model):
  images = [item for item in os.listdir(dir) if item.endswith('.jpg')]
  shuffle(images)

  for item in images:
    image = Image.open(os.path.join(dir, item))
    resized = image.resize((64, 64), Image.ANTIALIAS)
    h, w = image.size
    id = item.replace('.jpg', '')
    pixels = np.zeros((64, 64), dtype=float)
    for i in range(64):
      for j in range(64):
        pixels[i][j] = resized.getpixel((i, j))
    data[id] = pixels

data = dict()
resize('./images', data)

dir = './images'
