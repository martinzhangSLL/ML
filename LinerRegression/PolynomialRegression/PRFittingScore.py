import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(666)
x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)
y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=10)
x_train.shape

total_train_score=np.array([])
index=np.array([])

step=20
for i in range(2,100,step):
    poly_reg=Pipeline([
        ("poly",PolynomialFeatures(degree=i)),
        ("std_scale",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])
    train_score=[]
    index=np.append(index,i)
    for j in range(11,100):
        poly_reg.fit(x_train[:j],y_train[:j])
        y_train_predict=poly_reg.predict(x_train[:j])  
        train_score.append(r2_score(y_train[:j],y_train_predict))

    if(total_train_score.shape[0]==0):
        total_train_score=np.array(train_score)
    else:
        total_train_score=np.vstack((total_train_score,train_score))
    
for i in range(len(index)):
    plt.plot([(j-10) for j in range(11,100)],total_train_score[i],label='train'+str(index[i]))

plt.legend()
plt.show()