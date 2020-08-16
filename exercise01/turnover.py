import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('max_columns', 50)
train = pd.read_csv('train.csv')
train.head()
test = pd.read_csv('test.csv')
test.head()

train.info()

train['Over18'].value_counts()  # 只包含一种属性
train = train.drop(['EmployeeNumber', 'Over18'], axis=1)  # 删除无用特征

f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot("Gender", "Age", hue="Attrition", data=train, split=True, ax=ax[0])
ax[0].set_title('Gender and Age vs Attrtion');

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.countplot('JobLevel',hue='Attrition',data=train,ax=ax[0])
ax[0].set_title('JobLevel vs Attrition')
sns.countplot('Department',hue='Attrition',data=train,ax=ax[1])
ax[1].set_title('Department vs Attrition');

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.distplot(train[train['Attrition']==1].MonthlyIncome,ax=ax[0])
ax[0].set_title('MonthlyIncome vs Attrition=1')
sns.distplot(train[train['Attrition']==0].MonthlyIncome,ax=ax[1])
ax[1].set_title('MonthlyIncome vs Attrition=0');


# 筛选出字符型的变量
col = []
for i in train.columns:
    if train.loc[:, i].dtype == 'object':
        col.append(i)

# 导入相关模块进行数据类型转换
from sklearn.preprocessing import OrdinalEncoder

train.loc[:, col] = OrdinalEncoder().fit_transform(train.loc[:, col])
train.loc[:, col].head()

sns.heatmap(train.corr(),annot=True) #train.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,20);
plt.show()

X = train.loc[:,train.columns!='Attrition']
Y = train.loc[:,'Attrition']

X_onehot = X.drop(['Age','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole',
                   'YearsSinceLastPromotion', 'YearsWithCurrManager'],axis=1)

from sklearn.preprocessing import OneHotEncoder
X_onehot = OneHotEncoder().fit_transform(X_onehot)
X_onehot = X_onehot.toarray()

X_other = X.loc[:,['Age','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole',
                   'YearsSinceLastPromotion', 'YearsWithCurrManager']]
# X_other.shape
# X = pd.concat([pd.DataFrame(X_onehot),X_other],axis=1)
# X.shape
#
# from sklearn.model_selection import train_test_split
# Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=42)
# Xtrain.shape
#
# from sklearn.linear_model import LogisticRegression
# LR = LogisticRegression().fit(Xtrain,Ytrain)
# LR.score(Xtest,Ytest)

test = test.drop(['EmployeeNumber', 'Over18', 'MonthlyIncome'], axis=1)

# 筛选出字符型的变量
col = []
for i in test.columns:
    if test.loc[:, i].dtype == 'object':
        col.append(i)

from sklearn.preprocessing import OrdinalEncoder

test.loc[:, col] = OrdinalEncoder().fit_transform(test.loc[:, col])
test.loc[:, col].head()

test_onehot = test.drop(
    ['Age', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
     'YearsSinceLastPromotion', 'YearsWithCurrManager'], axis=1)

from sklearn.preprocessing import OneHotEncoder

test_onehot = OneHotEncoder().fit_transform(test_onehot)
test_onehot = test_onehot.toarray()

test_other = test.loc[:, ['Age', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
                          'YearsInCurrentRole',
                          'YearsSinceLastPromotion', 'YearsWithCurrManager']]
test_other.shape
X = pd.concat([pd.DataFrame(test_onehot), test_other], axis=1)
X.shape

solution = pd.DataFrame({'result': LR_o.predict(X)})
solution