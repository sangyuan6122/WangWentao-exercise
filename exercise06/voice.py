import pandas as pd

#数据加载
df = pd.read_csv('voice.csv')
print(df.head)
print(df.isnull().sum())
print(df.shape)
print('样本个数:{}'.format(df.shape[0]))
print('男性个数:{}'.format(df[df.label=='male'].shape[0]))
print('女性个数:{}'.format(df[df.label=='female'].shape[0]))

#分割特征和target
X=df.iloc[:, :-1]
# print(X.head)
y=df.iloc[:, -1]

#使用标签编码
from sklearn.preprocessing import LabelEncoder,StandardScaler
gender_encoder = LabelEncoder()
y=gender_encoder.fit_transform(y)
print(y)

scaler=StandardScaler()
X=scaler.fit_transform(X)
print(X)
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC
from sklearn import metrics
svc = SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('SVM 预测结果',y_pred)
print('SVM 准确率',metrics.accuracy_score(y_test,y_pred))