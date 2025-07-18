import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    #根据最小二乘法构造预测函数
    def fit(self, x_train, y_train):
        """根据训练数据集x_train, y_train训练模型"""
        #明确集合的维度是一维
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train."

        # 计算分子：Σ(xi - x̄)(yi - ȳ)
        # 使用向量化操作计算协方差的分子部分，.dot() 方法进行向量点积运算
        num = (x_train - x_train.mean()).dot(y_train - y_train.mean())
        
        # 计算分母：Σ(xi - x̄)²
        # 计算x的方差分子部分（即x偏差的平方和）
        d = (x_train - x_train.mean()).dot(x_train - x_train.mean())

        self.a_ = num / d
        self.b_ = y_train.mean() - self.a_ * x_train.mean()

        print(f"斜率a: {self.a_:.4f}, 截距b: {self.b_:.4f}")
        print(f"线性回归方程: y = {self.a_:.4f}x + {self.b_:.4f}")

        return self

    def _predict(self,x_single):
        """给定单个x，返回x的预测值y"""
        
        return self.a_ * x_single + self.b_
    
    def predict(self,x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""   
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        
        # 使用列表推导式对每个输入值进行预测
        # 调用私有方法_predict()计算每个x对应的y值
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def score(self, y_true, y_predict):
        return r2_score(y_true,y_predict)
    
    def mean_squared_error(self, y_true, y_predict):
        return mean_squared_error(y_true, y_predict)
    
    def mean_absolute_error(self, y_true, y_predict):
        return mean_absolute_error(y_true, y_predict)

    def __repr__(self):
        return f"SimpleLinearRegression(): y = {self.a_:.4f}x + {self.b_:.4f}"

if __name__ == "__main__":
    # 测试简单线性回归模型
    import matplotlib.pyplot as plt
    
    # 生成随机的线性数据
    np.random.seed(42)
    x = 2 * np.random.rand(100)
    y = 4 + 3 * x + np.random.randn(100)

    # 划分数据集为训练集和测试集
    x_train, x_test = x[:80], x[80:]
    y_train, y_test = y[:80], y[80:]

    # 创建简单线性回归模型实例
    lin_reg = SimpleLinearRegression()
    
    # 使用训练集数据训练模型
    lin_reg.fit(x_train, y_train)
    y_formular=lin_reg.predict(x_train)
    print(f"均方误差MSE: {lin_reg.mean_squared_error(y_train, y_formular):.4f}")
    print(f"均方根误差RMSE: {np.sqrt(lin_reg.mean_squared_error(y_train, y_formular)):.4f}")
    print(f"平均绝对误差MAE: {lin_reg.mean_absolute_error(y_train, y_formular):.4f}")
    print(f"R方值: {lin_reg.score(y_train, y_formular):.4f}")
    
    # 使用测试集数据进行预测
    y_predict = lin_reg.predict(x_test)
    print(f"均方误差MSE: {lin_reg.mean_squared_error(y_test, y_predict):.4f}")
    print(f"均方根误差RMSE: {np.sqrt(lin_reg.mean_squared_error(y_test, y_predict)):.4f}")
    print(f"平均绝对误差MAE: {lin_reg.mean_absolute_error(y_test, y_predict):.4f}")
    print(f"R方值: {lin_reg.score(y_test, y_predict):.4f}")
    
    # 打印模型参数
    print(lin_reg)

    plt.scatter(x_train, y_train,color='orange')
    plt.plot(x_train, y_formular, color='red')
    plt.axis([0, 2.5, 0, 13])
    plt.scatter(x_test, y_test, color='green')
    plt.scatter(x_test, y_predict, color='blue')
    plt.show()
    
    # 打印测试集的前5个预测值
    #print(y_predict[:5])