import numpy as np
# 导入必要的库
import numpy as np  # 添加numpy导入
import matplotlib.pyplot as plt  # 用于绘图

# 定义自变量x，创建一个包含1到10的numpy数组
# 这些是我们的输入数据点（特征值）
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 定义因变量y，对应x的输出值
# 这些是我们要预测的目标值
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# 绘制散点图，观察数据分布
plt.scatter(x, y)
plt.axis([0, 13, 0, 13])  # 设置坐标轴范围
plt.show()  # 显示图形

# 计算x和y的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 初始化分子和分母
up = 0    # 分子：Σ(xi-x̄)(yi-ȳ)
down = 0  # 分母：Σ(xi-x̄)²

# 使用最小二乘法计算线性回归参数
# 遍历每个数据点，计算回归系数
for x_i, y_i in zip(x, y):
    up += (x_i - x_mean) * (y_i - y_mean)    # 累加分子
    down += (x_i - x_mean) ** 2              # 累加分母

# 计算斜率a（回归系数）
a = up / down

# 计算截距b
# 使用公式：b = ȳ - a*x̄
b = y_mean - a * x_mean

# 输出线性回归方程的参数
# y = ax + b
print(f"斜率a: {a}, 截距b: {b}")
print(f"线性回归方程: y = {a:.4f}x + {b:.4f}")