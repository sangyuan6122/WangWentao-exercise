import torch
from torch import nn
from torch import optim
from model import TextRNN
from cnews_loader import read_vocab, read_category, process_file

#设置数据目标
train_file='cnews.train.small.txt'
test_file='cnews.test.txt'
val_file='cnews.val.txt'
vocab_file='cnews.vocab.txt'

#获取新闻的类别和对应的id的字典
categories,cat_to_id=read_category()
words,word_to_id=read_vocab(vocab_file)
vocab_size=len(words)
print(vocab_size)

x_train,y_train=process_file(train_file,word_to_id,cat_to_id,max_length=600)
x_val,y_val=process_file(val_file,word_to_id,cat_to_id,max_length=600)
#设置使用GPU
# cuda=torch.device('cuda')
x_train,y_train = torch.LongTensor(x_train),torch.LongTensor(y_train)
x_val,y_val = torch.LongTensor(x_val),torch.LongTensor(y_val)

import torch.utils.data as Data
train_dataset = Data.TensorDataset(x_train,y_train)
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=1000,shuffle=True)
val_dataset=Data.TensorDataset(x_val,y_val)
val_loader=Data.DataLoader(dataset=val_dataset,batch_size=1000,shuffle=True)

import numpy as np
# import pickle
# output=open('model.pkl','wb')
# pickle.dump(model,output)
def train():
    #创建模型
    model=TextRNN()
    #设置损失函数
    Loss=nn.MultiLabelSoftMarginLoss()
    optmizer=optim.Adam(model.parameters(),lr=0.001)
    best_val_acc=0
    for epoch in range(1000):
        print('epoch=',epoch)
        for step,(x_batch,y_batch) in enumerate(train_loader):
            x=x_batch
            y = y_batch
            out=model(x)
            loss=Loss(out,y)
            print('loss=',loss)
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
            accuracy=np.mean((torch.argmax(out,1)==torch.argmax(y,1)).cpu().numpy())
            print(accuracy)
        #对模型进行验证
        if(epoch+1) %5 ==0:
            for step,(x_batch,y_batch) in enumerate(val_loader):
                x=x_batch
                y=y_batch
                out=model(x)
                accuracy=np.mean((torch.argmax(out,1) == torch.argmax(y,1)).cpu().numpy())
                if accuracy>best_val_acc:
                    torch.save(model.state_dict(),'model_best.pkl')
                    best_val_acc=accuracy
                    print('model_best.pkl update')
                    print(accuracy)

process_file(train_file,word_to_id,cat_to_id,max_length=600)

train()