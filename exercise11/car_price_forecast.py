import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import utils

#导入数据集
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

utils.print_line('查看数据的形状（行数和列数）')
print('train_data shape :', train_data.shape) # (150000, 31)
print('test_data shape :', test_data.shape) # (50000, 30)

utils.print_line('数据概览')
print(train_data.head().append(train_data.tail()))
print(test_data.head().append(test_data.tail()))

utils.print_line('数据信息的查看 .info()可以看到每列的type，以及NAN缺失值的信息')
train_data.info()
utils.print_line('数据的统计信息概览')
print(train_data.describe())
utils.print_line('查看每列存在nan的情况')
print(train_data.isnull().sum())
#对nan进行可视化
missing = train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.show()
# 可视化缺失值,sample(250)表示抽取250个样本
msno.matrix(train_data.sample(250)) #
plt.show()

"""查看预测值的频数"""
train_data['price'].value_counts()

# 直方图可视化 自动划分10（默认值）个价格区间 统计每个区间的频数
plt.hist(train_data['price'], orientation='vertical', histtype='bar', color='red')
plt.show()

"""总体分布概况（无界约翰逊分布等）"""
import scipy.stats as st
y = train_data['price']

plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.show()
plt.subplot(1, 3, 2)
plt.title('normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.show()
plt.subplot(1, 3, 3)
plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
plt.show()

"""类别特征箱型图可视化"""
categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox',
                        'notRepairedDamage','regionCode', 'seller', 'offerType']
for c in categorical_features:
    train_data[c] = train_data[c].astype('category')
    if train_data[c].isnull().any():
        train_data[c] = train_data[c].cat.add_categories(['MISSING'])
        train_data[c] = train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxenplot(x=x, y=y)
    x = plt.xticks(rotation=90)

f = pd.melt(train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")
plt.show()
