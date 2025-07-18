import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from KNNClassifier import KNNClassifierCommon


#split the data into train set and test set
#给定一个数据集，根据设定的ratio。将数据集分割成一个训练集（1-ratio）和一个测试集（ratio）
def train_test_split(data,test_ratio=0.2,seed=None):

    assert 0.0<=test_ratio<=1.0,"test_ratio must be valid"
    assert data.shape[0]>0,"data must be valid"
   
    if seed:
        np.random.seed(seed)

    #生成一个打乱序号的集合，shuffle_indexes记录打乱的序号
    shuffled_indexes=np.random.permutation(len(data))

    test_size=int(len(data)*test_ratio)

    #取序号集的前len*ratio个作为测试集
    test_set=shuffled_indexes[:test_size]
    #取序号集的后len*（1-ratio）个作为训练集
    train_set=shuffled_indexes[test_size:]
    #返回训练集和测试集
    return data[train_set],data[test_set]

def test_iris():

    # 加载鸢尾花数据集
    # 该集合有特征数据和标签数据，通过对特征数据进行分类，预测标签数据
    iris = datasets.load_iris()
    x = iris.data      # 特征数据：花萼长度、花萼宽度、花瓣长度、花瓣宽度
    y = iris.target    # 标签数据：0-山鸢尾、1-变色鸢尾、2-维吉尼亚鸢尾
    
    # 将特征和标签合并为一个数组，方便后续数据划分
    cls = np.column_stack((x, y))
    
    # 划分训练集和测试集：80%训练，20%测试，设置随机种子保证结果可重现
    train_set, test_set = train_test_split(cls, test_ratio=0.2, seed=666)
    
    # 设置KNN算法的K值为5，即考虑5个最近邻
    k = 5
    knn_clf = KNNClassifierCommon(k)
    
    # 训练KNN分类器
    # train_set[:, :4] 取前4列作为训练特征
    # train_set[:, 4] 取第5列作为训练标签
    knn_clf.fit(train_set[:, :4], train_set[:, 4])

    # 准备测试数据
    test_data = test_set[:, :4]    # 测试集的特征数据
    test_target = test_set[:, 4]   # 测试集的真实标签

    # 使用训练好的模型对测试数据进行预测
    test_predict_target = knn_clf.predict(test_data)
    
    # 计算预测准确率
    correct_sum = np.sum(test_predict_target == test_target)  # 预测正确的样本数
    accuracy = correct_sum / len(test_target)                 # 准确率 = 正确数 / 总数
    
    # 输出评估结果
    print(f"corrct sum: {correct_sum}; total count:{len(test_target)};accuracy:{accuracy}")

def show_test_iris():

    # 加载鸢尾花数据集
    # 该集合有特征数据和标签数据，通过对特征数据进行分类，预测标签数据
    iris = datasets.load_iris()
    x = iris.data      # 特征数据：花萼长度、花萼宽度、花瓣长度、花瓣宽度
    y = iris.target    # 标签数据：0-山鸢尾、1-变色鸢尾、2-维吉尼亚鸢尾
    
    # 将特征和标签合并为一个数组，方便后续数据划分
    cls = np.column_stack((x, y))
    
    # 划分训练集和测试集：80%训练，20%测试，设置随机种子保证结果可重现
    train_set, test_set = train_test_split(cls, test_ratio=0.2, seed=666)
    
    # 设置KNN算法的K值为5，即考虑5个最近邻
    k=selectK(x,y)
    knn_clf = KNNClassifierCommon(k)
    
    # 训练KNN分类器
    # train_set[:, :4] 取前4列作为训练特征
    # train_set[:, 4] 取第5列作为训练标签
    knn_clf.fit(train_set[:, :4], train_set[:, 4])

    # 准备测试数据
    test_data = test_set[:, :4]    # 测试集的特征数据
    test_target = test_set[:, 4]   # 测试集的真实标签

    # 使用训练好的模型对测试数据进行预测
    test_predict_target = knn_clf.predict(test_data)
    
    # 计算预测准确率
    correct_sum = np.sum(test_predict_target == test_target)  # 预测正确的样本数
    accuracy = correct_sum / len(test_target)                 # 准确率 = 正确数 / 总数
    
    # 输出评估结果
    print(f"corrct sum: {correct_sum}; total count:{len(test_target)};accuracy:{accuracy}")


    # 可视化展示分类结果
    train_set_x_0=(train_set[train_set[:,4] == 0])[:,0]
    train_set_y_0=(train_set[train_set[:,4] == 0])[:,2]

    train_set_x_1=(train_set[train_set[:,4] == 1])[:,0]
    train_set_y_1=(train_set[train_set[:,4] == 1])[:,2]

    train_set_x_2=(train_set[train_set[:,4] == 2])[:,0]
    train_set_y_2=(train_set[train_set[:,4] == 2])[:,2]

    plt.scatter(train_set_x_0,train_set_y_0,c='r')
    plt.scatter(train_set_x_1,train_set_y_1,c='g')
    plt.scatter(train_set_x_2,train_set_y_2,c='b')

    test_set_x_0=(test_set[test_predict_target == 0])[:,0]
    test_set_y_0=(test_set[test_predict_target == 0])[:,2]

    test_set_x_1=(test_set[test_predict_target == 1])[:,0]
    test_set_y_1=(test_set[test_predict_target == 1])[:,2]

    test_set_x_2=(test_set[test_predict_target == 2])[:,0]
    test_set_y_2=(test_set[test_predict_target == 2])[:,2]

    plt.scatter(test_set_x_0,test_set_y_0,c='orange')
    plt.scatter(test_set_x_1,test_set_y_1,c='yellow')
    plt.scatter(test_set_x_2,test_set_y_2,c='purple')


    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('KNN Classification Results')
    plt.show()

# 选择最优K值的方法
# 通过交叉验证评估不同K值的性能，找到准确率最高的K值
def selectK(data, target):
    """
    使用交叉验证选择KNN算法的最优K值
    
    参数:
        data: 特征数据，形状为 (n_samples, n_features)
        target: 目标标签，形状为 (n_samples,)
    
    返回:
        int: 最优的K值
    """
    
    # 定义K值的搜索范围：从1到30
    # 通常K值不会设置得太大，避免过度平滑
    k_range = range(1, 31)
    
    # 存储每个K值对应的交叉验证准确率
    k_score = []
    
    # 遍历每个可能的K值
    for k in k_range:
        # 创建KNN分类器，设置邻居数为k
        # 使用sklearn的KNeighborsClassifier进行标准化实现
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # 使用10折交叉验证评估当前K值的性能
        # cv=10: 将数据分成10份，轮流用9份训练、1份测试
        # scoring='accuracy': 使用准确率作为评估指标
        scores = cross_val_score(knn, data, target, cv=10, scoring='accuracy')
        
        # 计算10次交叉验证结果的平均准确率
        # 并添加到k_score列表中
        k_score.append(scores.mean())
    
    # 绘制K值与准确率的关系图
    # 帮助可视化不同K值的性能表现
    plt.plot(k_range, k_score)
    plt.xlabel('Value of K for KNN')  # X轴标签：K值
    plt.ylabel('Cross-Validated Accuracy')  # Y轴标签：交叉验证准确率
    plt.show()
    
    # 找到准确率最高的K值
    # k_score.index(max(k_score)): 找到最大准确率的索引位置
    # +1: 因为k_range从1开始，而列表索引从0开始，所以需要加1
    return k_score.index(max(k_score)) + 1



if __name__ == "__main__":
    
    show_test_iris()
    
    


