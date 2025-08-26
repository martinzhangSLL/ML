"""
    测试逻辑回归模型的主函数
    使用鸢尾花数据集进行二分类任务演示
    """
    
from LR import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DrawDecisionBoundary import plot_decision_boundary



    # 加载鸢尾花数据集
iris=datasets.load_iris()
X=iris.data
y=iris.target
    
# 选择前两个类别进行二分类，只使用前两个特征便于可视化
X=X[y<2,:2]  # 只取前两个特征：萼片长度和萼片宽度
y=y[y<2]     # 只取前两个类别：setosa(0) 和 versicolor(1)
    
# 划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)

# 创建并训练逻辑回归模型
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
    
# 可视化数据分布
plot_decision_boundary(log_reg,axis=[4,7.5,1.5,4.5])
plt.scatter(X[y==0,0],X[y==0,1],color='red')    # setosa类别用红色
plt.scatter(X[y==1,0],X[y==1,1],color='blue')   # versicolor类别用蓝色
plt.show()

# 输出模型参数和性能
print(log_reg.coef_)                    # 特征系数
print(log_reg.intercept_)               # 截距项
print(log_reg.score(X_test,y_test))     # 测试集准确率
print(log_reg.predict_proba(X_test))    # 测试集预测概率
print(y_test)                           # 测试集真实标签