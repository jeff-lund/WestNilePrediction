import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def vectorize_codesum(cs):
    codesum = ['BR', 'RA', 'VCFG', 'TS',
               'DZ', 'FG+', 'GR', 'FU',
               'SN', 'TSRA', 'VCTS', 'SQ', 'FG',
               'BCFG', 'MIFG', 'HZ']
    vec = [0 for _ in range(len(codesum))]
    if cs == ' ':
        return np.array(vec)
    for code in cs.split(' '):
        if code == ' ' or code == '':
            continue
        vec[codesum.index(code)] = 1
    return np.array(vec)

def month(d):
    return int(d.split('-')[1])

def day(d):
    return int(d.split('-')[2])

def dataset(path='data'):
    codesum = ['BR', 'RA', 'VCFG', 'TS',
               'DZ', 'FG+', 'GR', 'FU',
               'SN', 'TSRA', 'VCTS', 'SQ', 'FG',
               'BCFG', 'MIFG', 'HZ']
    # read in csv files
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    weather = pd.read_csv(os.path.join(path, 'weather.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))

    # preprocessing
    y_train = train.WnvPresent.values
    x_train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis=1)
    x_test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis=1)

    x_train['month'] = x_train.Date.apply(month)
    x_train['day'] = x_train.Date.apply(day)
    x_test['month'] = x_test.Date.apply(month)
    x_test['day'] = x_test.Date.apply(day)

    # process weather data
    # replace text/missing values
    weather = weather.replace('M', -1)
    weather = weather.replace(['T', '  T', 'T ', ' T'], -1)
    weather = weather.replace('-', -1)
    #change codesum strings in to vectors
    code_vec = [vectorize_codesum(code) for code in weather['CodeSum']]
    for i, code in enumerate(codesum):
        temp = [v[i] for v in code_vec]
        weather[code] = temp
    #weather["CodeVec"] = code_vec
    weather = weather.drop('CodeSum', axis=1)
    # merge stations into single row
    w1 = weather[weather['Station']==1]
    w2 = weather[weather['Station']==2]
    w1 = w1.drop('Station', axis=1)
    w2 = w2.drop('Station', axis=1)
    weather = w1.merge(w2, on='Date')

    x_train = x_train.merge(weather, on='Date')
    x_test = x_test.merge(weather, on='Date')
    x_train = x_train.drop(['Date'], axis=1)
    x_test = x_test.drop(['Date'], axis=1)

    le = LabelEncoder()
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

    return x_train, y_train, x_test
