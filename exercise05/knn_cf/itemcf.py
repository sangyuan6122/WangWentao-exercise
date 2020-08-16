from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import evaluate, print_perf
# 数据读取
# reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
# data = Dataset.load_from_file('./ratings.csv', reader=reader)
data = Dataset.load_builtin('ml-100k')
# k折交叉验证(k=3)
data.split(n_folds=3)

trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})
algo.fit(trainset)

uid = str(196)
iid = str(302)

pred = algo.predict(uid, iid)
print(pred)

perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
#输出结果
print(perf)

perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)
