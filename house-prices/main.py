import pandas as pd
import numpy as np

# MSSubClass = dict([(20, 0)])
# fields = dict([('MSSubClass', MSSubClass), ()])

(train_filename, test_filename) = ('./data/train.csv', './data/test.csv')
train = pd.read_csv(train_filename, header='infer')
test = pd.read_csv(test_filename, header='infer')

def convert_features(df):
    df.pop('Id')

    # MSSubClass as str
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # MSZoning NA in pred. filling with most popular values
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

    # LotFrontage  NA in all. I suppose NA means 0
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

    # Alley  NA in all. NA means no access
    df['Alley'] = df['Alley'].fillna('NOACCESS')

    # Converting OverallCond to str
    df.OverallCond = df.OverallCond.astype(str)

    # MasVnrType NA in all. filling with most popular values
    df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
    # NA in all. NA means No basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('NoBSMT')

    # TotalBsmtSF  NA in pred. I suppose NA means 0
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)

    # Electrical NA in pred. filling with most popular values
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # KitchenAbvGr to categorical
    df['KitchenAbvGr'] = df['KitchenAbvGr'].astype(str)

    # KitchenQual NA in pred. filling with most popular values
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

    # FireplaceQu  NA in all. NA means No Fireplace
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFP')

    # GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
    for col in ('GarageType', 'GarageFinish', 'GarageQual'):
        df[col] = df[col].fillna('NoGRG')

    # GarageCars  NA in pred. I suppose NA means 0
    df['GarageCars'] = df['GarageCars'].fillna(0.0)

    # SaleType NA in pred. filling with most popular values
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

    # Year and Month to categorical
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    # Adding total sqfootage feature and removing Basement, 1st and 2nd floor df
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # df.drop(['MiscFeature', 'PoolQC', 'Fence', 'Alley' ], axis=1, inplace=True)

    df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

# ignore ['MiscFeature', 'PoolQC', 'Fence', 'FireplaceQu', 'Alley' ]
def preprocess(df):
    labels = df.pop('SalePrice')

    convert_features(df)

    numeric_features = df.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
    numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()

    return (df, labels)

(train_X, train_y) = preprocess(train)
print train_X.head()
# def process(input):
#     labels = input.pop('SalePrice')
#     input.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'], axis=1, inplace=True)
#     return (input, labels)

# train['Heating'].fillna(0, inplace=True)
# print train
