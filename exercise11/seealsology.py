# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
from ge import DeepWalk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#数据加载
df = pd.read_csv('./seealsology-data.tsv',sep='\t')
print(df)
#构造图
G = nx.from_pandas_edgelist(df,'source','target',edge_attr=True,create_using=nx.Graph())
print(G.nodes())
print(len(G.nodes))

#初始化模型，4线程
model=DeepWalk(G,walk_length=10,num_walks=5,workers=4)
#模型训练
model.train(window_size=4,iter=20)
#得到节点的embedding
embeddings=model.get_embeddings()
print(embeddings['critical illness insurance'])

#得到word2vec model
wv=model.w2v_model
print(wv.wv.similar_by_word('critical illness insurance'))

# 二维空间中绘制所选节点向量
def plot_nodes(word_list):
    X = []
    for item in word_list:
        X.append(embeddings[item])
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # 节点向量
    plt.figure(figsize=(12, 9))
    # 散点图
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(list(word_list)):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

plot_nodes(model.w2v_model.wv.vocab)