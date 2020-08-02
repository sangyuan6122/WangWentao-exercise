#加载停用词
with open('chinese_stopwords.txt','r',encoding='utf-8') as file:
    stopwords=[i[:-1] for i in file.readlines()]

import pandas as pd
#数据加载
news = pd.read_csv('sqlResult.csv',encoding='gb18030')
#文件尺寸
print(news.shape)
print(news.head())

#处理缺失值
news=news.dropna(subset=['content'])
print(news.shape)

import jieba
#分词
def split_text(text):
    text=text.replace(' ','').replace('\n','')
    text2=jieba.cut(text.strip())
    #去掉停用词
    result = ' '.join(w for w in text2 if w not in stopwords)
    return result

print(news.iloc[0].content)
print(split_text(news.iloc[0].content))

#对所有文本进行分词
corpus=list(map(split_text,[str(i) for i in news.content]))
print(corpus[0])
print(len(corpus))
import pickle
import os
if not os.path.exists('corpus.pkl'):
    #缓存到文件
    with open('corpus.pkl','wb') as file:
        pickle.dump(corpus,file)
else:
    #获得缓存
    with open('corpus.pkl','rb') as file:
        corpus=pickle.load(file)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#计算corpus中的TFIDF矩阵
countvectorizer=CountVectorizer(encoding='gb18030',min_df=0.015)
tfidftransformer=TfidfTransformer()
#先TF，在IDF->TFIDF
countvector = countvectorizer.fit_transform(corpus)
tfidf=tfidftransformer.fit_transform(countvector)

#标记是否为自己的新闻
label=list(map(lambda source:1 if '新华' in str(source) else 0,news.source))
#数据切分
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
X_train,X_test,y_train,y_test=train_test_split(tfidf.toarray(),label,test_size=0.3)
clf=MultinomialNB()
clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)
import numpy as np
#使用模型检测抄袭新闻，预测风格
#prediction代表风格预测，prediction=1代表风格预测为新华社
#labels代表实际值，如果labels=1说明实际为新华社
prediction=clf.predict(tfidf.toarray())
labels=np.array(label)
#compare_news_index有两列，prediction为预测风格，labels为实际
compare_news_index=pd.DataFrame({'prediction':prediction,'labels':labels})
copy_news_index=compare_news_index[(compare_news_index['prediction'] == 1)&(compare_news_index['labels'] == 0)].index
#实际为新华社的新闻
xinhuashe_news_index=compare_news_index[(compare_news_index['labels']==1)].index
print('可能为copy的新闻条数',len(copy_news_index))

#使用kneans对文中进行聚类
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
normalizer=Normalizer()
scaled_array=normalizer.fit_transform(tfidf.toarray())

#25是人工定义的桶个数，聚类个数
kmeans=KMeans(n_clusters=25)
k_labels=kmeans.fit_predict(scaled_array)

#创建id_class
id_class = {index:class_ for index,class_ in enumerate(k_labels)}
from  _collections import defaultdict
class_id = defaultdict(set)
for index,class_ in id_class.items():
    # 只统计新华社发布的class_id
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)
from sklearn.metrics.pairwise import cosine_similarity
#查找相似文本
def find_similar_text(cpindex,top=10):
    #只在新华社发布的文章中查找 class_id_key=class,value=id
    dist_dict= {i:cosine_similarity(tfidf[cpindex],tfidf[i]) for i in class_id[id_class[cpindex]]}
    #从大到小进行排序
    return sorted(dist_dict.items(),key=lambda  x:x[1][0],reverse=True)[:top]

cpindex=3352
similar_list=find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭:\n',news.iloc[cpindex].content)
#找一篇相似的原文
similar2=similar_list[0][0]
print('相似原文:\n',news.iloc[similar2].content)