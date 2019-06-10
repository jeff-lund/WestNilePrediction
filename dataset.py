import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

def year(d):
    return int(d.split('-')[0])

def month(d):
    return int(d.split('-')[1])

def day(d):
    return int(d.split('-')[2])

def date(d):
    return d.split('T')[0]

def present_to_int(d):
    w2v = {'positive' : 1, 'negative' : 0}
    return w2v[d]

def merge_test_keys(path='data'):
    # full Chicago WNV dataset
    full = pd.read_csv(open(os.path.join(path, 'west-nile-virus-wnv-mosquito-test-results.csv', 'rb')))
    # test data from kaggle competition
    test = pd.read_csv(open(os.path.join(path, 'data/test.csv', 'rb')))

    # preprocess full data set
    full['Date'] = full['TEST DATE'].apply(date)

    full['month'] = full.Date.apply(month)
    full['day'] = full.Date.apply(day)
    full = full.rename(index=str, columns={"SEASON YEAR" : "year"})
    test['month'] = test.Date.apply(month)
    test['day'] = test.Date.apply(day)
    test['year'] = test.Date.apply(year)
    full['WnvPresent'] = full.RESULT.apply(present_to_int)

    full = full.rename(index=str, columns={"TRAP" : 'Trap'})
    converted = pd.merge(test, full[['month', 'day', 'year', 'Trap', 'WnvPresent']], on=['month', 'day', 'year', 'Trap'])
    converted = converted.drop(['month', 'day', 'year'], axis=1)
    converted.to_csv(os.path.join(path, 'test_with_labels.csv'), index=False)

def adjust_species(mosquito):
    if mosquito ==  'UNSPECIFIED CULEX':
        return 'CULEX PIPIENS'
    return mosquito

def impute(col):
    for i in range(len(col)):
        if col[i] == np.nan:
            col[i] = float(col[i-1])
        else:
            col[i] = float(col[i])
    return col

def shuffle(X, Y):
    shuffle = np.arange(len(X))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    Y = Y[shuffle]
    return X, Y

def merge_test_keys():
    # full Chicago WNV dataset
    full = pd.read_csv(open('data/west-nile-virus-wnv-mosquito-test-results.csv', 'rb'))
    # test data from kaggle competition
    test = pd.read_csv(open('data/test.csv', 'rb'))

    # preprocess full data set
    full['Date'] = full['TEST DATE'].apply(date)

    full['month'] = full.Date.apply(month)
    full['day'] = full.Date.apply(day)
    full = full.rename(index=str, columns={"SEASON YEAR" : "year"})
    test['month'] = test.Date.apply(month)
    test['day'] = test.Date.apply(day)
    test['year'] = test.Date.apply(year)
    full['WnvPresent'] = full.RESULT.apply(present_to_int)

    full = full.rename(index=str, columns={"TRAP" : 'Trap'})
    converted = pd.merge(test, full[['month', 'day', 'year', 'Trap', 'WnvPresent']], on=['month', 'day', 'year', 'Trap'])
    converted = converted.drop(['month', 'day', 'year'], axis=1)
    converted.to_csv('data/test_with_labels.csv', index=False)

def dataset(path='data', pca=False, impute=False):
    le = LabelEncoder()
    codesum = ['BR', 'RA', 'VCFG', 'TS',
               'DZ', 'FG+', 'GR', 'FU',
               'SN', 'TSRA', 'VCTS', 'SQ', 'FG',
               'BCFG', 'MIFG', 'HZ']
    # read in csv files
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    weather = pd.read_csv(os.path.join(path, 'weather.csv'))
    test = pd.read_csv(os.path.join(path, 'test_with_labels.csv'))
    # preprocessing
    y_train = train.WnvPresent.values
    y_test = test.WnvPresent.values
    x_train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos', 'AddressAccuracy'], axis=1)
    x_test = test.drop(['Id', 'Address', 'AddressNumberAndStreet', 'WnvPresent', 'AddressAccuracy'], axis=1)

    x_train['month'] = x_train.Date.apply(month)
    x_train['day'] = x_train.Date.apply(day)
    x_test['month'] = x_test.Date.apply(month)
    x_test['day'] = x_test.Date.apply(day)
    x_test['Species'] = x_test.Species.apply(adjust_species)
    # process weather data
    # replace text/missing values

    weather = weather.replace(['T', '  T', 'T ', ' T'], 0.001)
    weather = weather[weather['Station']==1]
    weather = weather.drop(['Water1', 'SnowFall', 'Depth', 'CodeSum'] , axis=1)
    if impute:
        weather = weather.replace('M', np.nan)
        weather = weather.replace('-', np.nan)
        for key in weather:
            if key == 'Date' or key == 'station':
                continue
            weather[key] = weather[key].astype(float)
        weather = weather.interpolate()
    else:
        weather = weather.replace('M', 0)
        weather = weather.replace('-', 0)
    # merge stations into single row
    #w1 = weather[weather['Station']==1]
    #w2 = weather[weather['Station']==2]
    #w1 = w1.drop('Station', axis=1)
    #w2 = w2.drop('Station', axis=1)
    #weather = w1.merge(w2, on='Date')

    # uncomment to only use station 1


    x_train = x_train.merge(weather, on='Date')
    x_test = x_test.merge(weather, on='Date')
    x_train = x_train.drop(['Date'], axis=1)
    x_test = x_test.drop(['Date'], axis=1)

    le.fit(list(x_train['Species'].values) + list(x_test['Species'].values))
    x_train['Species'] = le.transform(x_train['Species'].values)
    x_test['Species'] = le.transform(x_test['Species'].values)

    le.fit(list(x_train['Street'].values) + list(x_test['Street'].values))
    x_train['Street'] = le.transform(x_train['Street'].values)
    x_test['Street'] = le.transform(x_test['Street'].values)

    le.fit(list(x_train['Trap'].values) + list(x_test['Trap'].values))
    x_train['Trap'] = le.transform(x_train['Trap'].values)
    x_test['Trap'] = le.transform(x_test['Trap'].values)

    x_train = x_train.to_numpy(float)
    x_test = x_test.to_numpy(float)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    if pca:
        print(len(x_train[0]))
        pca_model = PCA(.95)
        pca_model.fit(x_train)
        print(pca_model.n_components_)
        x_train = pca_model.transform(x_train)
        x_test = pca_model.transform(x_test)

    return x_train, y_train, x_test, y_test

def dataset2(path='data'):
    le = LabelEncoder()

    # read in csv files
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    weather = pd.read_csv(os.path.join(path, 'weather.csv'))
    #test = pd.read_csv(os.path.join(path, 'test.csv'))
    test = pd.read_csv(os.path.join(path, 'test_with_labels.csv'))
    # preprocessing
    y_train = train.WnvPresent.values
    y_test = test.WnvPresent.values

    x_train = train.drop(['Address', 'Species', 'Block', 'Street',
                          'Trap', 'AddressNumberAndStreet', 'AddressAccuracy',
                          'NumMosquitos', 'WnvPresent'], axis=1)

    x_test = test.drop(['Id', 'Address', 'AddressNumberAndStreet',
                        'WnvPresent', 'Block', 'Street',
                        'Species', 'Trap', 'AddressAccuracy' ], axis=1)

    weather = weather.drop(['Depth', 'SnowFall', 'Water1', 'CodeSum',
                            'Sunrise', 'Sunset', 'SeaLevel',
                            'ResultSpeed', 'ResultDir', 'AvgSpeed',
                            'PrecipTotal', 'Heat', 'Cool',
                            'Depart'], axis=1)

    x_train['month'] = x_train.Date.apply(month)
    x_train['day'] = x_train.Date.apply(day)
    x_test['month'] = x_test.Date.apply(month)
    x_test['day'] = x_test.Date.apply(day)

    # process weather data
    # replace text/missing values
    # replcaing trace values with small number
    weather = weather.replace(['T', '  T', 'T ', ' T'], 0.0001)
    # Replaces missing values with adjacent values
    weather = weather.replace('M', None)
    weather = weather.replace('-', None)

    # only use data from station 1
    weather = weather[weather['Station']==1]
    weather = weather.drop("Station", axis=1)

    x_train = x_train.merge(weather, on='Date')
    x_test = x_test.merge(weather, on='Date')
    x_train = x_train.drop(['Date'], axis=1)
    x_test = x_test.drop(['Date'], axis=1)

    x_train = x_train.to_numpy(float)
    x_test = x_test.to_numpy(float)
    x_train, y_train = (shuffle(x_train, y_train))
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    merge_test_keys()
