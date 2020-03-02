import os
import pandas as pd

dir = '/Users/chunyanwang/Documents/Christine/projects/data files/downloaded stock data/stocks daily data - tushare/'
files = os.listdir(dir)

data_all = pd.DataFrame()

for f in files:
    data = pd.read_csv(dir+f,header=0)
    if 'date' not in data.columns or len(f) < 10:
        continue
    data = data.set_index('date')
    data = data.sort_index()

    dates = list(data.index.values)

    num = len(data)
    if num > 200 #and 'industry profit rank' in data.columns and 'industry ave profit' in data.columns and \
        #data['industry profit rank'][:10].max() > 1:
        data = data.loc[data['over tendays vol'] == 1]
        #data1 = data[['up in 10 days','bigger vol support','vspre9daysmaxvol', 'continuous good up days',\
        #              'trend', 'index trend','today profit']]
        '''
        if data.iloc[0]['industry'] not in data.columns:
            print(data.iloc[0]['industry'] + ' not found')
            continue
        '''
        data['fivedays range before over'] = (data['close'].rolling(5).max()-data['close'].rolling(5).min())/data['close'].rolling(5).min
        
        data1 = data[['industry profit rank','industry profit rank in 3days','industry profit rank in 5days',
                      'industry ave profit','vsprevol','today up good','trend','index trend','next3days maxprofit',
                      data.iloc[0]['industry']]]
        data_all = pd.concat([data_all,data1])
        # prepare data_all data for model
        data_all.to_csv(
            '/Users/chunyanwang/Christine documents/projects/machine learning strategies/DecisionTree data.csv')
        print(len(data),len(data_all))
        print(f,' is done')
