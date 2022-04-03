from efficient_apriori import apriori
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

##导入数据并处理
path1 = r'D:\\course\\数据挖掘\Wine Reviews\\winemag-data_first150k.csv'
path2 = r'D:\\course\\数据挖掘\Wine Reviews\\winemag-data-130k-v2.csv'

data1 = read_csv(path1)
data2 = read_csv(path2)
#print(data1.columns)
#print(data2.columns)

#data1pre = data1[['country','province','points','price']]
#data2pre = data2[['country','province','points','price']]
data1pre = data1[['country','province']]
data2pre = data2[['country','province']]
datapre = data1pre.append(data2pre)
#print(datapre.shape)
datapre.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
#print(datapre.shape)
datapre = np.array(datapre)
datapre = datapre.tolist()

##频繁模式和关联规则处理
itemsets, rules = apriori(datapre, min_support=0.1,  min_confidence=0)
print("频繁项集：", itemsets)
print("关联规则：", rules)
it = ['US', 'California', 'France', 'Italy', 'California,US']
num = [116901, 80755, 43191, 43018, 80755]
plt.bar(it, num)
plt.xlabel("rules")
plt.ylabel("nums")
plt.show()
