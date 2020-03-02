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
from IPython.display import Image
import pydot

data = pd.read_csv('/Users/chunyanwang/Christine documents/projects/big vol before tp strategy/big vol before tp strategy all daily data1.csv',header=0)
data = data[data['buy']>0]
data = data.rename(columns={data.columns[0]:'date'})
data = data.set_index('date')
data['today high change'] = data['high']/data['prev close']-1
data['today low change'] = data['low']/data['prev close']-1
data['close to ma20'] = data['close']/data['ma20'] - 1
data['market value'] = data['volume'] * data['close']
data.to_csv('/Users/chunyanwang/Christine documents/projects/big vol before tp strategy/big vol before tp strategy all daily data.csv',index='date')

data1 = data[['market value', 'close to ma20','today profit','close_open per','today low change',
'next day open change','today open change', 'today high change', 'high_close per','close_low per',
'close']]
x = data1.values
y = data['total profits'].values

x_train, x_test, y_train, y_test= model_selection.train_test_split(x, y, test_size = 0.3, random_state = 0)

m1 = tree.DecisionTreeRegressor(max_features=11, max_depth=2,splitter='best')
m2 = tree.DecisionTreeRegressor(max_features=11,max_depth=8,splitter='best')
m3 = tree.DecisionTreeRegressor(max_features=11,max_depth=16,splitter='best')

m = m3.fit(x_train,y_train)
print(x_train,y_train)
dot_data = io.StringIO()

dot_data=tree.export_graphviz(m,
                    feature_names=data1.columns,
                    out_file=None,
                    filled=True,
                    rounded=True,
                    special_characters=True)

print(type(dot_data))
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTreeRegressor.png')
#graph.write_pdf('/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTreeRegressor.pdf')
#graph[0].write_png('/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTreeRegressor.png')
#Image(graph.create_png())

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)

print('---------',m1.feature_importances_)
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
plt.plot(y_test,y3,color='blue',label='depth=6',linewidth=2)
plt.plot(y_test,y2,color='green',label='depth=4',linewidth=2)
plt.plot(y_test,y1,color='red',label='depth=2',linewidth=2)
plt.xlabel('actual')
plt.ylabel('predict')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

plt.savefig('/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTreeRegressor1.png')