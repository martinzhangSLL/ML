import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 生成100个在[-3, 3]区间内均匀分布的随机数作为特征变量x
# np.random.uniform(low, high, size) 生成均匀分布的随机数
# 选择[-3, 3]区间是为了展示二次函数的完整形状（包括对称轴两侧）
x=np.random.uniform(-3,3,size=100)

# 将一维数组x转换为二维数组X，用于机器学习算法输入
# reshape(-1,1) 将形状从(100,)转换为(100,1)
# 大多数机器学习算法要求特征矩阵为二维格式
X=x.reshape(-1,1)

# 根据二次函数关系生成目标值y，并添加噪声
# y = x² + 2x + 1 是真实的二次函数关系
# np.random.normal(0,1,size=100) 添加均值为0、标准差为1的高斯噪声
# 噪声模拟真实数据中的测量误差和随机波动
y=x**2+2*x+1+np.random.normal(0,1,size=100)

# 尝试使用简单线性回归模型进行拟合
lin_reg=LinearRegression();
lin_reg.fit(X,y)
y_predict=lin_reg.predict(X)
print(f"Linear Regression R方值: {r2_score(y, y_predict):.4f}")

# 尝试使用多项式回归模型进行拟合
# 多项式回归是线性回归的一种扩展，它允许模型包含特征的多项式项
# 这里我们使用二次多项式（degree=2）
(X**2).shape
X2=np.hstack([X,X**2])
X2.shape
lin_reg2=LinearRegression();
lin_reg2.fit(X2,y)
y_predict2=lin_reg2.predict(X2)
print(f"Polinomial Regression R方值: {r2_score(y, y_predict2):.4f}")
print(f"Formular: y={lin_reg2.coef_[1]:.4f}x**2 +{lin_reg2.coef_[0]:.4f}x +{lin_reg2.intercept_:.4f}")


# 绘制散点图可视化生成的数据
# 横轴为特征x，纵轴为目标值y
# 可以观察到明显的二次函数（抛物线）模式
plt.scatter(x,y)
plt.plot(x,y_predict,color='r')
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='g')

# 显示图形
# 通过可视化可以直观地看到数据呈现二次函数的分布特征
plt.show()
