import array
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
# 生成100个在[0,2)范围内的随机数作为特征x
x=2*np.random.random(size=100)
# 根据线性关系y=3x+4生成目标值，并添加正态分布噪声
y=3*x+4+np.random.randn(100)

# 将一维数组x转换为二维数组X，用于机器学习算法输入
X=x.reshape(-1,1)
# 查看x的形状 (应该是(100,))
x.shape
# 查看y的形状 (应该是(100,))
y.shape

plt.scatter(x,y)
plt.show()

# 损失函数：计算均方误差(MSE)
# 就是J(theta)函数，一般采用MSE公式
def J(theta, X_b, y):
    try:
        # 计算预测值与真实值的差的平方和，再除以样本数量
        # y - X_b.dot(theta) 计算残差
        # ** 2 计算平方
        # np.sum() 求和
        # / len(X_b) 求平均
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except:
        # 如果计算出现异常（如数值溢出），返回无穷大
        return float('inf')

# 梯度函数：计算损失函数对每个参数的偏导数
def dj(theta, X_b,y):
    # 创建与theta同样长度的空数组存储梯度
    res=np.empty(len(theta))
    # 计算截距项(theta[0])的梯度
    # 对theta[0]的偏导数 = 2 * sum(X_b.dot(theta) - y) / m
    res[0]=np.sum(X_b.dot(theta)-y)
    # 计算其他参数(theta[1], theta[2], ...)的梯度
    for i in range(1,len(theta)):
        # 对theta[i]的偏导数 = 2 * sum((X_b.dot(theta) - y) * X_b[:, i]) / m
        res[i]=(X_b.dot(theta)-y).dot(X_b[:,i])
    # 乘以2/m得到完整的梯度公式
    return res*2/len(X_b)

# 梯度下降算法主函数
# https://zhuanlan.zhihu.com/p/77253076 算法
def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    """
    参数说明：
    X_b: 增广特征矩阵（包含截距列）
    y: 目标值
    initial_theta: 初始参数值
    eta: 学习率
    n_iters: 最大迭代次数，默认10000次
    epsilon: 收敛阈值，默认1e-8
    """
    theta = initial_theta  # 初始化参数
    i_iter = 0  # 迭代计数器
    
    # 开始迭代优化
    while i_iter < n_iters:
        # 计算当前参数下的梯度
        gradient = dj(theta, X_b, y)
        # 保存上一次的参数值，用于判断收敛
        last_theta = theta
        # 梯度下降更新参数：theta = theta - 学习率 * 梯度
        theta = theta - eta * gradient
        
        # 判断是否收敛：如果损失函数变化小于阈值则停止
        if (abs(J(last_theta, X_b, y) - J(theta, X_b, y)) < epsilon):
            break
        i_iter += 1  # 迭代次数加1
    
    return theta  # 返回优化后的参数

# 构建增广特征矩阵：在X前面添加全1列作为截距项
# np.ones((len(x),1)) 创建100x1的全1矩阵
# np.hstack() 水平拼接，得到[1, x]的矩阵形式
X_b=np.hstack([np.ones((len(X),1)),X])

# 初始化参数：创建与特征数量相同的零向量
# X_b.shape[1] = 2 (截距项 + 1个特征)
# 本实例是一个线性方程，所以是y=a1+a2x的格式，所以参数数量是2，所以要构建一个二维的矩阵[a1,a2]这种格式
initial_theta=np.zeros(X_b.shape[1])

# 设置学习率
eta=0.01

# 执行梯度下降算法，获得最优参数
theta=gradient_descent(X_b,y,initial_theta,eta)

# 输出最终的参数结果
# theta[0] 是截距项，theta[1] 是斜率
print(theta)