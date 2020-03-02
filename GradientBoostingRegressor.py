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

data_all = pd.read_csv('/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTree data.csv')
data_all['trend'] = data_all['trend'].apply(trend_to_num)
data_all['index trend'] = data_all['index trend'].apply(trend_to_num)

data_all = data_all.replace([np.inf,-np.inf],np.nan)
data_all = data_all.dropna()

#data_all = pd.read_csv('/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTree data - small.csv',header=0)
x = data_all[['industry profit rank','industry profit rank in 3days','industry profit rank in 5days',\
                      'industry profit rank in 10days','industry ave profit','trend','index trend']].values.round(4)
y = data_all['next3days maxprofit'].values.round(4)
'''
# classify y result to y<2%: 1, 2%<=y<5%: 2; 5%=<y<=8%: 3; y>= 8%: 4.
y_classified = []
for i in y:
    if i < 0:
        y_classified.append(0)
    elif i < 0.02:
        y_classified.append(1)
    elif i <= 0.05:
        y_classified.append(2)
    elif i <= 0.08:
        y_classified.append(3)
    else:
        y_classified.append(4)
print(y_classified)
'''

x_train, x_test, y_train, y_test= model_selection.train_test_split(x, y, test_size = 0.3, random_state = 0)

m1 = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=2)
m2 = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=4)
m3 = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=6)

m = m2.fit(x_train,y_train)

''' ensemble tree can not be graphed
dot_data = io.StringIO()
tree.export_graphviz(m,
                    out_file=dot_data,
                    feature_names=['industry profit rank','industry profit rank in 3days','industry profit rank in 5days',\
                      'industry profit rank in 10days','industry ave profit','trend','index trend'],
                    class_names=['amazing','good','medium','low','bad'],
                    filled=True,
                    rounded=True,
                    special_characters=True)
print(type(dot_data))
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('GradientBoostingRegressor.png')
'''
m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)

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
plt.title('Decision Tree Classifier')
plt.legend()
plt.show()

plt.savefig('/Users/chunyanwang/Christine documents/projects/machine learning strategies/GradientBoostingRegressor.png')