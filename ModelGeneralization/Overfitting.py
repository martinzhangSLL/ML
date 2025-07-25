import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 展示一个过拟合的例子

def PolynomialRegression(degree):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])

np.random.seed(666)
x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)
y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

poly_reg100=PolynomialRegression(100)
poly_reg100.fit(X,y)
y_predict=poly_reg100.predict(X)
score=mean_squared_error(y,y_predict)


print(score)

# 系数之间的差异超过了10个数量级，说明模型过拟合
print(poly_reg100.named_steps["lin_reg"].coef_)

x_plot=np.linspace(-3,3,100).reshape(100,1)
y_plot=poly_reg100.predict(x_plot)

plt.scatter(x,y)
plt.plot(x_plot[:,0],y_plot,color="red")
plt.axis([-3,3,-1,10])
plt.show()
