import torch
from sklearn.datasets import load_boston

#数据加载
boston = load_boston()
X = boston['data']
y=boston['target']
print(X)
print(X.shape)
print(y)

#将y进行转变形状
y=y.reshape(-1,1)
print(y)

from sklearn.preprocessing import MinMaxScaler
#数据规范化
ss_input=MinMaxScaler()
print(X)

X=ss_input.fit_transform(X)
print(X)

#numpy =>tensor
X=torch.from_numpy(X).type(torch.FloatTensor)
y=torch.from_numpy(y).type(torch.FloatTensor)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.25)

from torch import nn

#构造网络
model=nn.Sequential(
    nn.Linear(13,16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

#定义优化器、损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

#训练
max_epoch=300
iter_loss=[]
for i in range(max_epoch):
    #前向传播
    y_pred=model(X)
    #计算loss
    loss=criterion(y_pred,y)
    iter_loss.append(loss.item())
    print(loss.item())
    #清空之前的梯度
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #参数更新
    optimizer.step()

#测试
output=model(test_x)
predict_list=output.detach().numpy()
print(predict_list)

import numpy as np
import matplotlib.pyplot as plt
#绘制不同iteration的loss
x=np.arange(max_epoch)
y=np.array(iter_loss)
plt.plot(x,y)
plt.title('loss Value in 300 epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Loss Value')
plt.show()

#查看真实值与预测值的散点图
x=np.arange(test_x.shape[0])
y1=np.array(predict_list)
y2=np.array(test_y)
line1=plt.scatter(x,y1,c='red')
line2=plt.scatter(x,y2,c='yellow')
plt.legend([line1,line2],['predict','real'])
plt.title('Prediction VS Real')
plt.ylabel('House Price')
plt.show()