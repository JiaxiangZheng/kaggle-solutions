import pandas as pd
import numpy as np
from pandas import read_csv, DataFrame

print('installed pandas version is %s' % pd.__version__)

names = ['jiaxzheng', 'mliu']
ages  = [27, 30]
data = list(zip(names, ages))
df = DataFrame(data=data, columns=['name', 'age'])

df = read_csv('./data/train.csv')
# show summary info of the data frame
df.info()
print('****** df.head *****')
print(df.head())
print('****** df.tail *****')
print(df.tail())

for column in df:
  dfc = df[column]
  if dfc.dtype in ('int64', 'float64'):
    dfc.fillna(dfc.mean(), inplace=True)
  else:
    dfc.fillna('unkown', inplace=True)

df.info()
# show the columns of the data frame
df.columns

# methods of column values
df.Heating.unique()
print(df.Heating.describe())

# fillna
print df.FireplaceQu.unique()
print df.FireplaceQu.fillna('', inplace=True)
print df.FireplaceQu.unique()
print df.FireplaceQu.describe()

# data frame concat
df1 = DataFrame(data=np.random.rand(3, 4), columns=['A', 'B', 'C', 'D'])
df2 = DataFrame(data=np.random.rand(3, 4), columns=['A', 'B', 'D', 'E'])
df_concat = pd.concat([df1, df2], keys=['train', 'test'])
df_concat.drop(['D'], axis=1, inplace=True)
df_concat.C.fillna(0, inplace=True)
df_concat.E.fillna(0.5, inplace=True)
print(df_concat)
