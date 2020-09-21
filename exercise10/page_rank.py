import numpy as np
#有向图结构
a = np.array([
      [0, 0, 0, 1, 0, 0],
      [1, 0, 0, 0, 1, 0],
      [0, 1, 0, 1, 1, 0],
      [1, 0, 0, 0, 0, 1],
      [1, 0, 1, 1, 0, 0],
      [1, 0, 0, 0, 0, 0]],dtype=float)
# a_leak = np.array([[0, 0, 0, 1/2],
# 				   [0, 0, 0, 1/2],
# 				   [0, 1, 0, 0],
# 				   [0, 0, 1, 0]])
#
# a_sink = np.array([[0, 0, 0, 0],
# 				   [1/2, 0, 0, 1],
# 				   [0, 1, 1, 0],
# 				   [1/2, 0, 0, 0]])
#转移矩阵
def transPre(data):
    b = np.transpose(data) #把矩阵转置
    c = np.zeros((a.shape),dtype=float)
    #把所有的元素重新分配
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c[i][j] = data[i][j] / (b[j].sum())

    return c
a=transPre(a)
print('转移矩阵:')
print(transPre(a))
b = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
w = b

#简单模型
def work(a, w):
	for i in range(100):
		w = np.dot(a, w)
		print(w)

#随机浏览模型
def random_work(a, w, n):
	d = 0.85
	for i in range(100):
		w = (1-d)/n + d*np.dot(a, w)
		print(w)

work(a, w)
random_work(a, w, 6)
# random_work(a_leak, w, 4)
# random_work(a_sink, w, 4)