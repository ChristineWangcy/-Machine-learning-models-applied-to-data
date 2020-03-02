import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import pydotplus
import pandas as pd

import io
import sklearn.preprocessing as skp
from sklearn import tree
from sklearn.metrics import r2_score
import graphviz
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn import model_selection
from sklearn import ensemble

def trend_to_num(item):
    if item == 'up':
        return 6
    elif item == 'upph':
        return 5
    elif item == 'tz':
        return 4
    elif item == 'ft':
        return 3
    elif item == 'downph':
        return 2
    elif item == 'down':
        return 1
    else:
        return 0

data_all = pd.read_csv('/Users/chunyanwang/Christine documents/projects/big vol before tp strategy/big vol before tp strategy all daily data.csv')
print(data_all.columns())
exit()
data_all = data_all.replace([np.inf,-np.inf],np.nan)
data_all = data_all.dropna()

x = data_all[['']].values.round(4)
y = data_all['total profit'].values.round(4)


x_train, x_test, y_train, y_test= model_selection.train_test_split(x, y, test_size = 0.3, random_state = 0)

m1 = ensemble.RandomForestRegressor(max_depth=2,random_state=0)
m2 = ensemble.RandomForestRegressor(max_depth=4,random_state=0)
m3 = ensemble.RandomForestRegressor(max_depth=6,random_state=0)

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
print(m1.feature_importances_)
print(m2.feature_importances_)
print(m3.feature_importances_)

y1 = m1.predict(x_test)
y2 = m2.predict(x_test)
y3 = m3.predict(x_test)
y_true = y_test
y_pred = y1
print(r2_score(y_true,y_pred))

y_pred = y2
print(r2_score(y_true,y_pred))

y_pred = y3
print(r2_score(y_true,y_pred))

plt.figure()
plt.plot(y_test,y3,color='blue',linewidth=2)
plt.xlabel('target')
plt.ylabel('actual')
plt.title('RandomForestRegressor')
plt.legend()
plt.show()

plt.savefig('/Users/chunyanwang/Christine documents/projects/machine learning strategies/RandomForestRegressor.png')