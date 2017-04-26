import os
import numpy as np
import pandas as pd
from scipy import signal
from PIL import Image

def load_image(path):
    return np.asarray(Image.open(path))/255.0

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_image(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, 11)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 0.1

    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, inp, 1.0)

def validate(source_dir, target_dir):
  mse = 0.0
  count = 0
  for path in os.listdir(source_dir):
    if path.endswith('.png'):
      inp_path = os.path.join(dir, path)
      inp = load_image(inp_path)
      out = denoise_image(inp)
      real = load_image(os.path.join(target_dir, path))

      height, width = inp.shape
      count += width * height

      mse += np.sum((real - out) * (real - out))
      print path, np.sum((real - out) * (real - out))

  print mse, count
  print np.sqrt(mse / count)

def submit(test_dir):
  ids = []
  values = []
  for path in os.listdir(test_dir):
    print path
    if path.endswith('.png'):
      inp_path = os.path.join(test_dir, path)
      inp = load_image(inp_path)
      out = denoise_image(inp)

      # save(os.path.join(test_dir, 'cleaned_' + path), out)
      id = path.replace('.png', '')
      row, height = out.shape
      for i in range(1, row + 1):
        for j in range(1, height + 1):
          ids += [id + '_' + str(i) + '_' + str(j)]
          values += [out[i-1][j-1]]

  df = pd.DataFrame(data=list(zip(ids, values)), columns=['id', 'value'])
  df.to_csv('submit.csv', index=False)

# validate('./data/train', './data/train_cleaned')
submit('./data/test/')