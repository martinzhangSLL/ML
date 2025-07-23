import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None
        self._theta=None

    #根据最小二乘法构造预测函数
    def fit_normal(self, x_train, y_train):
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

    def fit_gd(self,x_train,y_train,eta=0.01):
        """
        使用梯度下降算法训练简单线性回归模型
        
        参数:
        x_train: 训练集特征数据（一维数组）
        y_train: 训练集目标值（一维数组）
        eta: 学习率，默认0.01
        
        返回:
        self: 返回训练后的模型实例，支持链式调用
        """
        # 验证输入数据的维度一致性
        assert x_train.shape[0]==y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train."
        
        # 定义损失函数（均方误差MSE）
        def J(theta, X_b,y):
            """
            计算损失函数值
            
            参数:
            theta: 参数向量 [截距, 斜率]
            X_b: 增广特征矩阵（包含截距列）
            y: 真实目标值
            
            返回:
            损失函数值（均方误差）
            """
            try:
                # 计算预测值与真实值的均方误差
                # X_b.dot(theta) 计算预测值
                # (y - X_b.dot(theta)) ** 2 计算残差的平方
                # np.sum() 求和，/ len(X_b) 求平均
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                # 如果计算过程中出现数值异常（如溢出），返回无穷大
                return float('inf')

        # 定义梯度函数（损失函数的偏导数）
        def dj(theta, X_b,y):
            """
            计算损失函数对参数的梯度（偏导数）
            
            参数:
            theta: 当前参数向量 [截距, 斜率]
            X_b: 增广特征矩阵
            y: 真实目标值
            
            返回:
            梯度向量，包含对每个参数的偏导数
            """
            # 创建与参数向量同样长度的梯度向量
            res=np.empty(len(theta))
            
            # 计算对截距项（theta[0]）的偏导数
            # ∂J/∂θ₀ = (2/m) * Σ(ŷᵢ - yᵢ)
            # 其中 ŷᵢ = X_b.dot(theta) 是预测值
            res[0]=np.sum(X_b.dot(theta)-y)
            
            # 计算对其他参数（theta[1], theta[2], ...）的偏导数
            for i in range(1,len(theta)):
                # ∂J/∂θᵢ = (2/m) * Σ((ŷᵢ - yᵢ) * xᵢⱼ)
                # X_b[:,i] 表示第i列特征值
                res[i]=(X_b.dot(theta)-y).dot(X_b[:,i])
            
            # 乘以 2/m 得到完整的梯度公式
            return res*2/len(X_b)

        # 定义梯度下降主算法
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """
            梯度下降算法实现
            
            参数说明：
            X_b: 增广特征矩阵（包含截距列）
            y: 目标值
            initial_theta: 初始参数值
            eta: 学习率
            n_iters: 最大迭代次数，默认10000次
            epsilon: 收敛阈值，默认1e-8
            
            返回:
            优化后的参数向量
            """
            theta = initial_theta  # 初始化参数
            i_iter = 0  # 迭代计数器
            
            # 开始迭代优化过程
            while i_iter < n_iters:
                # 计算当前参数下的梯度
                gradient = dj(theta, X_b, y)
                
                # 保存当前参数，用于收敛性判断
                last_theta = theta
                
                # 梯度下降更新规则：θ = θ - η * ∇J(θ)
                # eta是学习率，控制每次更新的步长
                theta = theta - eta * gradient
                
                # 检查收敛条件：如果损失函数变化小于阈值则停止迭代
                # 这避免了不必要的计算，提高效率
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                    
                i_iter += 1  # 迭代次数递增
            
            return theta  # 返回优化后的参数
        
        # 数据预处理：将一维特征转换为二维矩阵
        # reshape(-1,1) 将 (n,) 转换为 (n,1)
        x_train_b=x_train.reshape(-1,1)
        
        # 构建增广特征矩阵：在特征矩阵前添加全1列作为截距项
        # np.ones((len(x_train),1)) 创建全1列向量
        # np.hstack() 水平拼接，得到 [1, x] 的矩阵形式
        X_b=np.hstack([np.ones((len(x_train),1)),x_train_b])
        
        # 初始化参数向量：创建零向量
        # X_b.shape[1] = 2（截距项 + 1个特征）
        initial_theta=np.zeros(X_b.shape[1])
        
        # 执行梯度下降算法，获得最优参数
        self._theta=gradient_descent(X_b,y_train,initial_theta,eta)
        
        # 提取参数：theta[1]是斜率，theta[0]是截距
        self.a_=self._theta[1]  # 斜率参数
        self.b_=self._theta[0]  # 截距参数
        
        # 返回self支持链式调用，如：model.fit_gd(x,y).predict(x_test)
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
    lin_reg.fit_normal(x_train, y_train)
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


    # 创建简单线性回归模型实例
    dg_reg = SimpleLinearRegression()
    
    # 使用训练集数据训练模型
    dg_reg.fit_gd(x_train, y_train)
    y_formular=dg_reg.predict(x_train)
    print(f"均方误差MSE: {dg_reg.mean_squared_error(y_train, y_formular):.4f}")
    print(f"均方根误差RMSE: {np.sqrt(dg_reg.mean_squared_error(y_train, y_formular)):.4f}")
    print(f"平均绝对误差MAE: {dg_reg.mean_absolute_error(y_train, y_formular):.4f}")
    print(f"R方值: {dg_reg.score(y_train, y_formular):.4f}")    
    
    # 使用测试集数据进行预测
    y_predict = dg_reg.predict(x_test)  
    print(f"均方误差MSE: {dg_reg.mean_squared_error(y_test, y_predict):.4f}")
    print(f"均方根误差RMSE: {np.sqrt(dg_reg.mean_squared_error(y_test, y_predict)):.4f}")
    print(f"平均绝对误差MAE: {dg_reg.mean_absolute_error(y_test, y_predict):.4f}")
    print(f"R方值: {dg_reg.score(y_test, y_predict):.4f}")
    
    # 打印模型参数
    print(dg_reg)   

    plt.clf()
    plt.scatter(x_train, y_train,color='orange')
    plt.plot(x_train, y_formular, color='red')
    plt.axis([0, 2.5, 0, 13])
    plt.scatter(x_test, y_test, color='green')
    plt.scatter(x_test, y_predict, color='blue')
    plt.show()
    
    print('End')
    # 打印测试集的前5个预测值
    #print(y_predict[:5])