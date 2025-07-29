import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def plot_model(model,x_test=None,y_test=None, color='r'):

    x_plot=None
    y_plot=None
    if x_test is not None and y_test is not None:
        x_plot=x_test
        y_plot_origin=y_test
        y_plot=model.predict(x_plot)
        # 计算均方误差，正则化就是解决均方误差问题，越小越好
        mse=mean_squared_error(y_plot_origin,y_plot)
        print("MSE:",mse)
    else:
        x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
        y_plot = model.predict(x_plot)
    
    plt.scatter(x, y)
    plt.plot(np.sort(x_plot[:,0]), y_plot[np.argsort(x_plot[:,0])], color=color)
    plt.axis([-3, 3, 0, 6])
    

def generate_data():
    np.random.seed(666)
    x=np.random.uniform(-3,3,size=100)
    y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)
    x=x.reshape(-1,1)
    return x,y

def LassoRegression(degree,alpha):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("ridge_reg",Lasso(alpha=alpha))
    ])


def PolynomiaRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scale', StandardScaler()),
        ('lin_reg', LinearRegression()),
    ])

x,y=generate_data()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)

# No Regularization
poly_reg=PolynomiaRegression(20)
poly_reg.fit(x_train,y_train)
plot_model(poly_reg,x_test,y_test)

# 系数次在1-5之间，比较高
# 一次线性回归使用20次的复杂回归，出现过拟合
print(poly_reg.named_steps['lin_reg'].coef_)

# Regularization
# 正则化，alpha=0.0001，正则化项的权重，alpha越大，正则化项的权重越大，模型越简单
# score明显下降
lasso_reg=LassoRegression(20,alpha=0.0001)
lasso_reg.fit(x_train,y_train)
plot_model(lasso_reg,x_test,y_test,color='b')

# 正则化，alpha=1，正则化项的权重，alpha越大，正则化项的权重越大，模型越简单
# score明显下降
lasso_reg=LassoRegression(20,alpha=1)
lasso_reg.fit(x_train,y_train)
plot_model(lasso_reg,x_test,y_test,color='g')

# 正则化，alpha=100，正则化项的权重，alpha越大，正则化项的权重越大，模型越简单
# score开始上升，出现欠拟合情况
lasso_reg=LassoRegression(20,alpha=100)
lasso_reg.fit(x_train,y_train)
plot_model(lasso_reg,x_test,y_test,color='y')

plt.show()
