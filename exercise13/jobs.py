import pandas as pd

content = pd.read_excel('jobs_4k.xls')
print(content)
position_names=content['positionName'].tolist()
print('职位总量:'+str(len(position_names)))
print('职位种类:'+str(len(set(position_names))))

skill_lables=content['skillLables'].tolist()
print('技能数:'+str(len(skill_lables)))

from collections import defaultdict
skill_position_graph=defaultdict(list)
for p,s in zip(position_names,skill_lables):
    skill_position_graph[p]+=eval(s)
print(skill_position_graph)

import random
import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph(skill_position_graph)
#随机30工作岗位可视化
sample_nodes=random.sample(position_names,k=30)
sample_nodes_connections=sample_nodes
for p, skills in skill_position_graph.items():
    if p in sample_nodes:
        sample_nodes_connections+=skills
#抽取G节点为子图
sample_graph=G.subgraph(sample_nodes_connections)
plt.figure(figsize=(50,30))
pos=nx.spring_layout(sample_graph,k=1)
nx.draw(sample_graph,pos,with_labels=True,node_size=30,font_size=10)
#解决中文乱码
plt.rcParams['font.sans-serif']=['SimHei']
plt.show()

#核心能力、职位按影响力排序
pr=nx.pagerank(G,alpha=0.9)
ranked_position_and_ability=sorted([(name,value) for name,value in pr.items()],key=lambda x: x[1],reverse=True)
print(ranked_position_and_ability)

#取出特征X(不含salary)
X_content=content.drop(['salary'],axis=1)
#取出Target
target=content['salary'].tolist()
print(target)
#将X_content内容拼接为字符串
X_content['merged']=X_content.apply(lambda  x:''.join(str(x)),axis=1)
#转list
X_string=X_content['merged'].tolist()
print(X_string[0])

import jieba

#合并
def get_one_row_job_string(x_string_row):
    job_string=''
    for i,element in enumerate(x_string_row.split('\n')):
        if len(element.split()) == 2:
            _,value=element.split()
            if i==0:
                continue
            job_string+=value
    return job_string

import re
def token(string):
    return re.findall('\w+',string)
cutted_X=[]
for i,row in enumerate(X_string):
    job_string=get_one_row_job_string(row)
    cutted_X.append(' '.join(jieba.cut(''.join(token(job_string)))))

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(cutted_X)
print(X[0])
print(target[:5])

import numpy as np
#平均值
targer_numical=[np.mean(list(map(float,re.findall('\d+',s)))) for s in target]
print(targer_numical[0:5])

#KNN回归
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)
y=targer_numical
model.fit(X,y)
print(model.score)

def predict_by_label(test_string,model):
    test_words=list(jieba.cut(test_string))
    test_vec=vectorizer.transform(test_words)
    predict_y=model.predict(test_vec)
    return predict_y[0]
test1='测试 北京 3年 专科'
test2='测试 北京 4年 专科'
test3='算法 北京 4年 本科'
test4='UI 北京 4年 本科'
print(test1,predict_by_label(test1,model))
print(test2,predict_by_label(test2,model))
print(test3,predict_by_label(test3,model))
print(test4,predict_by_label(test4,model))
sentences=["广州Java本科3年掌握大数据",
           "沈阳Java硕士3年掌握大数据",
           "兰州Java本科10年掌握大数据",
           "北京算法硕士3年掌握图像识别"]
for p in sentences:
    print('{}薪酬预测{}'.format(p,predict_by_label(p,model)))