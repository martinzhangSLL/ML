import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    """
    逻辑回归分类器实现
    
    使用梯度下降算法训练逻辑回归模型，适用于二分类问题
    基于sigmoid函数将线性回归的输出映射到[0,1]区间作为概率
    """

    def __init__(self):
        """
        初始化逻辑回归模型
        
        属性:
        coef_: 特征系数（权重）
        intercept_: 截距项
        _theta: 完整的参数向量[截距, 系数1, 系数2, ...]
        """
        self.coef_=None
        self.intercept_=None
        self._theta=None
    
    def _sigmoid(self,t):
        """
        Sigmoid激活函数
        
        将任意实数映射到(0,1)区间，用于计算概率
        公式: σ(t) = 1 / (1 + e^(-t))
        
        参数:
        t: 输入值，可以是标量或数组
        
        返回:
        sigmoid函数的输出值，范围在(0,1)之间
        """
        return 1 / (1 + np.exp(-t))

    def fit(self,X_train,y_train,eta=0.01,n_iters=1e4):
        """
        使用梯度下降算法训练逻辑回归模型
        
        参数:
        X_train: 训练集特征矩阵，形状为(n_samples, n_features)
        y_train: 训练集标签向量，形状为(n_samples,)，值为0或1
        eta: 学习率，默认0.01
        n_iters: 最大迭代次数，默认10000
        
        返回:
        self: 训练后的模型实例，支持链式调用
        """
        # 验证输入数据的维度一致性
        assert X_train.shape[0]==y_train.shape[0],\
        "The size of X_train must be equal to the size of y_train"

        def J(theta,X_b,y):
            """
            逻辑回归的损失函数（对数似然损失）
            
            公式: J(θ) = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
            其中 ŷ = sigmoid(X·θ) 是预测概率
            
            参数:
            theta: 参数向量
            X_b: 增广特征矩阵（包含截距列）
            y: 真实标签
            
            返回:
            损失函数值，值越小表示模型拟合越好
            """
            # 计算预测概率
            y_hat=self._sigmoid(X_b.dot(theta))
            try:
                # 计算对数似然损失
                # -y*log(y_hat) 是正样本的损失
                # -(1-y)*log(1-y_hat) 是负样本的损失
                return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/len(y)
            except:
                # 如果计算过程中出现数值异常（如log(0)），返回无穷大
                return float('inf')
        
        def dJ(theta,X_b,y):
            """
            损失函数的梯度（偏导数向量）
            
            逻辑回归梯度的向量化形式:
            ∇J(θ) = 1/m * X^T * (sigmoid(X·θ) - y)
            
            参数:
            theta: 当前参数向量
            X_b: 增广特征矩阵
            y: 真实标签
            
            返回:
            梯度向量，指示参数更新的方向和大小
            """
            # X_b.T.dot() 实现矩阵乘法 X^T * (ŷ - y)
            # sigmoid(X_b.dot(theta)) - y 计算预测误差
            return X_b.T.dot(self._sigmoid(X_b.dot(theta))-y)/len(X_b)

        def gradient_descent(X_b,y,initial_theta,eta=0.01,n_iters=1e4,epsilon=1e-8):
            """
            梯度下降算法实现
            
            通过迭代更新参数来最小化损失函数
            更新规则: θ = θ - η * ∇J(θ)
            
            参数:
            X_b: 增广特征矩阵
            y: 目标标签
            initial_theta: 初始参数向量
            eta: 学习率
            n_iters: 最大迭代次数
            epsilon: 收敛阈值
            
            返回:
            优化后的参数向量
            """
            theta=initial_theta
            i_iter=0

            # 迭代优化过程
            while i_iter<n_iters:
                # 计算当前梯度
                gradient=dJ(theta,X_b,y)
                # 更新参数：θ = θ - η * ∇J(θ)
                theta=theta-eta*gradient
                # 检查收敛条件：损失函数变化小于阈值
                if abs(J(theta,X_b,y)-J(theta-eta*gradient,X_b,y))<epsilon:
                    break
                i_iter+=1
            
            return theta
        
        # 构建增广特征矩阵：在X_train前添加全1列作为截距项
        X_b=np.hstack([np.ones((len(X_train),1)),X_train])
        # 初始化参数为零向量
        initial_theta=np.zeros(X_b.shape[1])
        # 执行梯度下降优化
        self._theta=gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        # 提取截距和系数
        self.intercept_=self._theta[0]  # 截距项
        self.coef_=self._theta[1:]      # 特征系数
        return self

    def predict_proba(self,X_predict):
        """
        预测样本属于正类的概率
        
        参数:
        X_predict: 待预测的特征矩阵，形状为(n_samples, n_features)
        
        返回:
        概率数组，每个元素表示对应样本属于正类(y=1)的概率
        """
        # 验证模型已训练
        assert self.coef_ is not None and self.intercept_ is not None,\
        "must fit before predict!"
        # 验证特征维度一致性
        assert X_predict.shape[1]==len(self.coef_),\
        "the feature number of X_predict must be equal to X_train"

        # 构建增广特征矩阵
        X_b=np.hstack([np.ones((len(X_predict),1)),X_predict])
        # 计算并返回预测概率：σ(X·θ)
        return self._sigmoid(X_b.dot(self._theta))
    
    def predict(self,X_predict):
        """
        预测样本的类别标签
        
        使用0.5作为分类阈值：
        - 概率 >= 0.5 预测为正类(1)
        - 概率 < 0.5 预测为负类(0)
        
        参数:
        X_predict: 待预测的特征矩阵
        
        返回:
        预测标签数组，元素为0或1
        """
        # 验证模型已训练
        assert self.coef_ is not None and self.intercept_ is not None,\
        "must fit before predict!"
        # 验证特征维度一致性
        assert X_predict.shape[1]==len(self.coef_),\
        "the feature number of X_predict must be equal to X_train"

        # 获取预测概率
        y_hat=self.predict_proba(X_predict)
        # 应用0.5阈值进行二分类决策
        y_hat=np.array(y_hat>=0.5,dtype='int')
        return y_hat
    
    def score(self,X_test,y_test):
        """
        计算模型在测试集上的准确率
        
        参数:
        X_test: 测试集特征矩阵
        y_test: 测试集真实标签
        
        返回:
        准确率分数，范围[0,1]，值越大表示性能越好
        """
        y_predict=self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        """
        返回模型的字符串表示
        """
        return "LogisticRegression()"
    




