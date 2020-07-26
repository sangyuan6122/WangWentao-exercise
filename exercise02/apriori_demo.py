import pandas as pd
import time

# 数据加载
dataset = pd.read_csv('./Market_Basket_Optimisation.csv',header=None)
column_size=dataset.shape[1]
transactions=[]
for i in range(0,dataset.shape[0]):
    temp=[]
    for j in range(column_size):
        if str(dataset.values[i,j])!='nan':
            temp.append(str(dataset.values[i,j]))
    transactions.append(temp)
from efficient_apriori import apriori
itemsets,rules=apriori(transactions,min_support=0.02,min_confidence=0.3)
print('频繁项集:',itemsets)
print('关联规则:',rules)
