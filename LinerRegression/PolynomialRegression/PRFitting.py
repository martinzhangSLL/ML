import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Prepare Data
np.random.seed(666)
x=np.random.uniform(-3,3,size=100)  
X=x.reshape(-1,1)
y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

plt.scatter(x,y)

# 使用一次函数进行拟合
# 该种情况模型无法拟合二次函数的非线性关系
# 预测结果与真实值之间的差异较大
# 属于欠拟合
lin_reg=LinearRegression()
lin_reg.fit(X,y)
y_predict=lin_reg.predict(X)
print(f"Linear Regression R方值: {r2_score(y, y_predict):.4f}")
print(f"Formular: y={lin_reg.coef_[0]:.4f}x +{lin_reg.intercept_:.4f}")
print(f"系数: {lin_reg.coef_}")
print(f"截距: {lin_reg.intercept_}")
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')

# 使用二次函数进行拟合
# 预测结果与真实值之间的差异较小
# 属于拟合正常
poly_reg2=Pipeline([
    ("poly",PolynomialFeatures(degree=2)),
    ("lin_reg",LinearRegression())
])
poly_reg2.fit(X,y)
y_predict2=poly_reg2.predict(X)
print(f"Polinomial Regression R方值: {r2_score(y, y_predict2):.4f}")
print(f"Formular: y={poly_reg2.named_steps['lin_reg'].coef_[2]:.4f}x**2 +{poly_reg2.named_steps['lin_reg'].coef_[1]:.4f}x +{poly_reg2.named_steps['lin_reg'].intercept_:.4f}")
print(f"系数: {poly_reg2.named_steps['lin_reg'].coef_}")
print(f"截距: {poly_reg2.named_steps['lin_reg'].intercept_}")
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='g')

# 使用10次函数进行拟合
# 预测结果与真实值之间的差异较小
# 属于拟合正常
poly_reg10=Pipeline([
    ("poly",PolynomialFeatures(degree=10)),
    ("std_scale",StandardScaler()),
    ("lin_reg",LinearRegression())
])
poly_reg10.fit(X,y)
y_predict10=poly_reg10.predict(X)
print(f"Polinomial Regression R方值: {r2_score(y, y_predict10):.4f}")
print(f"系数: {poly_reg10.named_steps['lin_reg'].coef_}")
print(f"截距: {poly_reg10.named_steps['lin_reg'].intercept_}")
plt.plot(np.sort(x), y_predict10[np.argsort(x)], color='y')

# 使用100次函数进行拟合
# 预测结果与真实值之间的差异开始变大，而且震荡较大
# 属于过拟合
poly_reg100=Pipeline([
    ("poly",PolynomialFeatures(degree=100)),
    ("std_scale",StandardScaler()),
    ("lin_reg",LinearRegression())
])
poly_reg100.fit(X,y)
y_predict100=poly_reg100.predict(X)
print(f"Polinomial Regression R方值: {r2_score(y, y_predict100):.4f}")
print(f"系数: {poly_reg100.named_steps['lin_reg'].coef_}")
print(f"截距: {poly_reg100.named_steps['lin_reg'].intercept_}")
plt.plot(np.sort(x), y_predict100[np.argsort(x)], color='r')

plt.show()