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
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('/Users/chunyanwang/Christine documents/projects/big vol before tp strategy/big vol before tp strategy all daily data1.csv',header=0)
data = data[data['buy']>0]
data = data.rename(columns={data.columns[0]:'date'})
data = data.set_index('date')
data['today high change'] = data['high']/data['prev close']-1
data['today low change'] = data['low']/data['prev close']-1
data['close to ma20'] = data['close']/data['ma20'] - 1
data['market value'] = data['volume'] * data['close']
data.to_csv('/Users/chunyanwang/Christine documents/projects/big vol before tp strategy/big vol before tp strategy all daily data.csv',index='date')

data1 = data[['market value', 'close to ma20','today profit','today low change',
'next day open change','today open change', 'today high change']]
x = data1.values

data['total profits class'] = 0
data.loc[data['total profits'] > 0.02,'total profits class'] = 2
data.loc[(data['total profits'] <= 0.02) & (data['total profits']>0),'total profits class'] = 1
data.loc[data['total profits'] <= 0,'total profits class'] = 0

y = data['total profits class'].values

x_train, x_test, y_train, y_test= model_selection.train_test_split(x, y, test_size = 0.3, random_state = 0)

m = GaussianNB()
m.fit(x_train,y_train)

y = m.predict(x_test)

y_true = y_test
y_pred = y
print(r2_score(y_true,y_pred))

plt.figure()
plt.plot(y_true,y_pred,color='red',label = 'GaussianNB',linewidth=2)
plt.xlabel('actual')
plt.ylabel('predict')
plt.title('GaussianNB')
plt.legend()
plt.show()

plt.savefig('/Users/chunyanwang/Christine documents/projects/machine learning strategies/GaussianNB.png')