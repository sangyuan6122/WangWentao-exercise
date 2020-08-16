import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.width', 1000)
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(len(train))
print(len(test))
print(train.head())
print(test.head())

train.info()

id_col = 'user_id'
target_col = 'Attrition'

digital_cols = ['Age', 'DailyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
category_cols = ['BusinessTravel', 'Department',  'Education', 'EducationField',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel','DistanceFromHome',
                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime',
                'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'PerformanceRating', 'TrainingTimesLastYear','WorkLifeBalance' ]

# Credits to https://www.kaggle.com/a763337092/lr-baseline-for-bi-class#Data-process
# For categorical data
for col in category_cols:
    nunique_tr = train[col].nunique()
    nunique_te = test[col].nunique()
    na_tr = len(train.loc[train[col].isna()]) / len(train)
    na_te = len(test.loc[test[col].isna()]) / len(test)
    print(f'Col name:{col:30}\tunique cate num in train:{nunique_tr:5}\tunique cate num in train:{nunique_te:5}\tnull sample in train:{na_tr:.2f}\tnull sample in test:{na_te:.2f}')

# For numerical data
for col in digital_cols:
    min_tr = train[col].min()
    max_tr = train[col].max()
    mean_tr = train[col].mean()
    median_tr = train[col].median()
    std_tr = train[col].std()
    x = ['min', 'mean', 'median', 'std', 'max']
    y = [min_tr, mean_tr, median_tr, std_tr, max_tr]

    min_te = test[col].min()
    max_te = test[col].max()
    mean_te = test[col].mean()
    median_te = test[col].median()
    std_te = test[col].std()
    x = ['min', 'mean', 'median', 'std', 'max']
    y = [min_tr, mean_tr, median_tr, std_tr, max_tr]

    na_tr = len(train.loc[train[col].isna()]) / len(train)
    na_te = len(test.loc[test[col].isna()]) / len(test)
    print(f'\tIn train data:\tnan sample rate:{na_tr:.2f}\t')
    print(f'\tIn test data\tnan sample rate:{na_te:.2f}\t')
plt.bar(x, y)
plt.title(col)
plt.show()

#age and attrition
plt.figure(figsize=(4,3))
print(train['Attrition'])
sns.barplot(x='Attrition', y='Age', data = train , palette = 'Set2')
#plt.show()

figure, ax = plt.subplots(figsize=(10, 10))
data = pd.concat([train.drop(['user_id','Attrition','EmployeeNumber','EmployeeCount', 'Over18','StandardHours'],axis = 1), test]).corr() ** 2
#data = np.tril(data, k=-1)
data[data==0] = np.nan
sns.heatmap(np.sqrt(data), annot=False, cmap='viridis', ax=ax)
plt.show()
print(type(data))