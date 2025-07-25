import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 加载手写数字识别数据集
# digits数据集包含8x8像素的手写数字图像(0-9)，共1797个样本
digits=datasets.load_digits()

# 提取特征数据：每个样本是64维向量(8x8像素展平)
# x.shape = (1797, 64)，每行代表一个手写数字图像的像素值
x=digits.data

# 提取目标标签：数字类别(0-9)
# y.shape = (1797,)，每个元素是对应图像的真实数字标签
y=digits.target

# 划分训练集和测试集
# test_size=0.4 表示40%作为测试集，60%作为训练集
# random_state=666 确保每次运行时数据划分结果一致，便于结果复现
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=666)

# Use Simple Validation

# 初始化最佳参数记录变量
# best_score: 记录目前找到的最高准确率
# best_p: 记录对应的最佳距离度量参数(闵可夫斯基距离的p值)
# best_k: 记录对应的最佳邻居数量
best_score, best_p, best_k=0,0,0

# 网格搜索：遍历不同的超参数组合寻找最优配置
# 外层循环：遍历邻居数量k，范围从2到10
# k 判断标准，取最小距离的k个样本，取这k个样本中出现次数最多的类别作为预测结果
for k in range(2,11):
    # 内层循环：遍历距离度量参数p，范围从1到5
    # p=1: 曼哈顿距离, p=2: 欧几里得距离, p>2: 闵可夫斯基距离
    for p in range(1,6):
        # 创建KNN分类器实例
        # weights="distance": 使用距离加权，距离越近的邻居权重越大
        # n_neighbors=k: 设置邻居数量
        # p=p: 设置闵可夫斯基距离的参数
        knn_clf=KNeighborsClassifier(weights="distance",n_neighbors=k,p=p)
        
        # 使用训练集训练模型
        knn_clf.fit(x_train,y_train)
        
        # 在测试集上评估模型性能，获得准确率分数
        score=knn_clf.score(x_test,y_test)
        
        # 如果当前组合的性能超过历史最佳，则更新最佳参数
        if score>best_score:
            best_score,best_k,best_p=score,k,p

# 输出最佳参数组合和对应的准确率
print("Best K:", best_k)
print("Best P:", best_p)
print("Best Score:", best_score)

# Use Cross Validation

best_score, best_p, best_k=0,0,0

for k in range(2,11):
    for p in range(1,6):
        knn_clf=KNeighborsClassifier(weights="distance",n_neighbors=k,p=p)
        scores=cross_val_score(knn_clf,x_train,y_train)
        score=np.mean(scores)
        if score>best_score:
            best_score,best_k,best_p=score,k,p

print("Best K:", best_k)
print("Best P:", best_p)
print("Best Score:", best_score)

