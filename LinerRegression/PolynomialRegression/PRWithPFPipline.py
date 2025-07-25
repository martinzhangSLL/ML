import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# 使用和不使用 StandardScaler 会导致系数不一致，但这 并不意味着模型性能不同
# 系数的差异主要是由于 StandardScaler 对特征进行了标准化处理，
# 而 LinearRegression 模型在进行拟合时，会根据标准化后的特征和目标值进行计算。
# 因此，系数的差异是由于标准化处理导致的，而不是模型性能的差异。
# - 1. 系数不同是正常的 - 它们作用在不同尺度的特征上
# - 2. 预测结果相同 - 这才是模型的真正目标
poly_reg=Pipeline([
    ("poly",PolynomialFeatures(degree=2)),
    ("std_scale",StandardScaler()),
    ("lin_reg",LinearRegression())
])

poly_reg.fit(X,y)
y_predict=poly_reg.predict(X)
print(f"Polinomial Regression R方值: {r2_score(y, y_predict):.4f}")
print(f"Formular: y={poly_reg.named_steps['lin_reg'].coef_[2]:.4f}x**2 +{poly_reg.named_steps['lin_reg'].coef_[1]:.4f}x +{poly_reg.named_steps['lin_reg'].intercept_:.4f}")
print(f"系数: {poly_reg.named_steps['lin_reg'].coef_}")
print(f"截距: {poly_reg.named_steps['lin_reg'].intercept_}")



# 绘制散点图可视化生成的数据
# 横轴为特征x，纵轴为目标值y
# 可以观察到明显的二次函数（抛物线）模式
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],color='g')

# 显示图形
# 通过可视化可以直观地看到数据呈现二次函数的分布特征
plt.show()
