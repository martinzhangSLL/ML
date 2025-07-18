import numpy as np
from sklearn.metrics import r2_score

class MultiLinearRegression:
    
    def __init__(self):
        """初始化MultiLinearRegression模型"""
        #系数初始化，coefficient
        self.coef_ = None
        #截距初始化 intercept
        self.interception_ = None
        self._theta = None

    def fit_normal(self,x_train,y_train):

        """根据训练数据集x_train,y_train训练MultiLinearRegression模型"""
        assert x_train.shape[0] == y_train.shape[0],\
            "the size of x_train must be equal to the size of y_train"
    
        
        # 正规化
        # 构造设计矩阵X：在原始特征矩阵x_train前添加一列全1向量
        # np.ones((len(x_train),1)): 创建一个形状为(样本数, 1)的全1列向量，用于表示截距项
        # np.hstack: 水平拼接数组，将全1列向量与原始特征矩阵x_train合并
        # 结果X的形状为(样本数, 特征数+1)，第一列为1，后续列为原始特征
        X = np.hstack([np.ones((len(x_train), 1)), x_train])
        
        # 使用正规方程求解线性回归参数θ
        # 正规方程公式：θ = (X^T * X)^(-1) * X^T * y
        # X.T: X的转置矩阵
        # X.T.dot(X): 计算X^T * X
        # np.linalg.inv(): 计算矩阵的逆
        # .dot(X.T).dot(y_train): 依次计算 * X^T * y_train
        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)
        
        # 提取截距项：θ的第一个元素对应截距（常数项）
        # 因为设计矩阵X的第一列是全1向量
        self.interception_ = self._theta[0]
        
        # 提取系数向量：θ的第二个元素开始对应各特征的系数
        # self._theta[1:] 表示从索引1开始到末尾的所有元素
        self.coef_ = self._theta[1:]
        
        # 返回自身，支持链式调用
        return self
    
    def predict(self,x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        # 断言：x_predict的特征数必须等于模型的特征数
        # x_predict.shape[1] 表示特征数
        # self.coef_.shape[0] 表示模型的特征数
        # 两者必须相等，否则抛出异常
        assert x_predict.shape[1] == self.coef_.shape[0],\
            "the feature number of x_predict must be equal to the feature number of model"
        # 计算预测值向量：y_predict = X_predict * θ
        # X_predict: 待预测数据集，形状为 (样本数, 特征数)
        # θ: 模型参数向量，形状为 (特征数+1,)
        # np.dot(X_predict, self.coef_): 计算 X_predict * θ
        # self.interception_: 模型的截距项
        # 结果 y_predict 是一个向量，每个元素表示对应样本的预测值
        # 预测值 = 截距项 + 特征1 * 系数1 + 特征2 * 系数2 + ... + 特征n * 系数n
        # 其中，特征1, 特征2, ..., 特征n 是样本的特征值
        # 系数1, 系数2, ..., 系数n 是模型的参数，用于描述各特征对预测值的影响
        # 截距项 是模型的常数项，用于描述样本的平均预测值
        # 预测值向量 y_predict 是一个向量，每个元素表示对应样本的预测值
        # 预测值 = 截距项 + 特征1 * 系数1 + 特征2 * 系数2 + ... + 特征n * 系数n
        # 其中，特征1, 特征2, ..., 特征n 是样本的特征值
        # 系数1, 系数2, ..., 系数n 是模型的参数，用于描述各特征对预测值的影响
        # 截距项 是模型的常数项，用于描述样本的平均预测值
        # 预测值向量 y_predict 是一个向量，每个元素表示对应样本的预测值
        X=np.hstack([np.ones((len(x_predict),1)),x_predict])
        return X.dot(self._theta)
    
    def score(self,x_test,y_test):
        """根据测试数据集x_test,y_test确定当前模型的准确度"""
        # 计算预测值向量 y_predict
        # x_test 是待测试数据集，形状为 (测试样本数, 特征数)
        # self.coef_ 是模型的系数向量，形状为 (特征数,)
        # self.interception_ 是模型的截距项
        # np.dot(x_test, self.coef_) 计算 x_test * self.coef_
        # 结果是一个向量，每个元素表示对应样本的预测值
        # 预测值 = 截距项 + 特征1 * 系数1 + 特征2 * 系数2 + ... + 特征n * 系数n
        y_predict=self.predict(x_test)
        print("y_test:",y_test)
        print("y_predict:",y_predict)
        return r2_score(y_test,y_predict)
    

def clean_boston_data(boston_data):
    """
    专门处理Boston数据集的数据清理
    """
    # 确保输入是numpy数组
    if not isinstance(boston_data, np.ndarray):
        boston_data = np.array(boston_data)
    
    print(f"原始数据形状: {boston_data.shape}")
    print(f"原始数据类型: {boston_data.dtype}")
    
    # 如果是object类型，需要逐列检查
    if boston_data.dtype == 'object':
        numeric_cols = []
        for i in range(boston_data.shape[1]):
            col = boston_data[:, i]
            try:
                # 尝试转换为float
                numeric_col = np.array(col, dtype=np.float64)
                # 检查是否有有效的数值
                if not np.all(np.isnan(numeric_col)):
                    numeric_cols.append(numeric_col)
                else:
                    print(f"列 {i} 全部为NaN，已剔除")
            except (ValueError, TypeError) as e:
                print(f"列 {i} 无法转换为数值类型: {e}")
                continue
        
        if numeric_cols:
            # 重新组合数值列
            cleaned_data = np.column_stack(numeric_cols)
            print(f"清理后数据形状: {cleaned_data.shape}")
            return cleaned_data
        else:
            raise ValueError("没有找到有效的数值列")
    else:
        # 如果已经是数值类型，直接返回
        return boston_data

# 在主函数中使用
if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml
    from sklearn.linear_model import LinearRegression
    
    try:
        # 加载数据
        boston = fetch_openml(name='boston', version=1, as_frame=False)
        
        # 清理特征数据
        x_raw = boston.data
        y_raw = boston.target
        
        print("开始清理特征数据...")
        x = clean_boston_data(x_raw)
        
        # 清理目标数据
        print("开始清理目标数据...")
        try:
            y = np.array(y_raw, dtype=np.float64)
            # 移除NaN值
            valid_y_mask = ~np.isnan(y)
            y = y[valid_y_mask]
            x = x[valid_y_mask]  # 保持x和y的对应关系
        except Exception as e:
            print(f"目标数据清理失败: {e}")
            raise
        
        print(f"最终数据形状 - X: {x.shape}, Y: {y.shape}")
        
        # 过滤数据
        mask = y < 25
        X = x[mask]
        Y = y[mask]
        
        print(f"过滤后数据形状 - X: {X.shape}, Y: {Y.shape}")
        
        # 继续训练模型
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        #使用自己实现的多特征线性回归模型
        #multi_linear_reg = MultiLinearRegression()
        #multi_linear_reg.fit_normal(X_train, y_train)
        
        #print("Coefficients:", multi_linear_reg.coef_)
        #print("Interception:", multi_linear_reg.interception_)
       # print("R2 Score:", multi_linear_reg.score(X_test, y_test))

        #使用sklearn自带的多特征线性回归模型
        multi_linear_reg=LinearRegression()
        multi_linear_reg.fit(X_train, y_train)
        
        print("Coefficients:", multi_linear_reg.coef_)
        print("Interception:", multi_linear_reg.intercept_)
        print("R2 Score:", multi_linear_reg.score(X_test, y_test))
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
