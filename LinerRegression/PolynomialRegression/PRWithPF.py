import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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

# 创建多项式特征生成器，设置最高次数为2
# PolynomialFeatures会生成所有次数不超过degree的多项式特征
# degree=2 表示生成常数项、一次项和二次项特征
poly=PolynomialFeatures(degree=2)

# 拟合多项式特征生成器到输入数据X
# fit()方法学习输入数据的结构，确定需要生成哪些特征
# 对于一个特征的情况，会准备生成 [1, x, x²] 这样的特征组合
poly.fit(X)

# 将原始特征X转换为多项式特征矩阵
# 原始X的形状是(100, 1)，包含一个特征x
# 转换后X_poly包含：[1, x, x²] 三列特征
# 即：常数项(截距)、原始特征x、x的平方项
X_poly=poly.transform(X)

# 查看转换后的特征矩阵形状
# 预期结果：(100, 3) - 100个样本，3个特征列
# 第0列：全为1的常数项
# 第1列：原始特征x
# 第2列：x的平方项x²
X_poly.shape

lin_reg=LinearRegression();
lin_reg.fit(X_poly,y)
y_predict=lin_reg.predict(X_poly)
print(f"Polinomial Regression R方值: {r2_score(y, y_predict):.4f}")
print(f"Formular: y={lin_reg.coef_[2]:.4f}x**2 +{lin_reg.coef_[1]:.4f}x +{lin_reg.intercept_:.4f}")
print(f"系数: {lin_reg.coef_}")
print(f"截距: {lin_reg.intercept_}")



# 绘制散点图可视化生成的数据
# 横轴为特征x，纵轴为目标值y
# 可以观察到明显的二次函数（抛物线）模式
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],color='g')

# 显示图形
# 通过可视化可以直观地看到数据呈现二次函数的分布特征
plt.show()
