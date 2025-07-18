import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets

from KNNClassifier import KNNClassifier

#split the data into train set and test set
def train_test_split(data,test_ratio=0.2,seed=None):

    assert 0.0<=test_ratio<=1.0,"test_ratio must be valid"
    assert data.shape[0]>0,"data must be valid"
   
    if seed:
        np.random.seed(seed)

    shuffled_indexes=np.random.permutation(len(data))

    test_size=int(len(data)*test_ratio)
    test_set=shuffled_indexes[:test_size]
    train_set=shuffled_indexes[test_size:]

    return data[train_set],data[test_set]


if __name__ == "__main__":
    
    
    np.random.seed(42)
    x_cls1 = 1+2*np.random.rand(20)
    y_cls1 = 1+2*np.random.rand(20)
    cls1=np.column_stack((x_cls1,y_cls1))

    np.random.seed(42)
    x_cls2 = 3+2*np.random.rand(20)
    y_cls2 = 3+np.random.rand(20)
    cls2=np.column_stack((x_cls2,y_cls2))
    

    k=5
    knn_clf=KNNClassifier(k)
    knn_clf.setClass(cls1,cls2)

    new_spot1=np.array([1.5,2.5])
    new_spot2=np.array([3.5,3.5])
    
    vote_result1=knn_clf.predict(new_spot1)
    vote_result2=knn_clf.predict(new_spot2)

    print(f"The spot:{new_spot1} belongs to class:{vote_result1}")
    print(f"The spot:{new_spot2} belongs to class:{vote_result2}")

    plt.scatter(x_cls1, y_cls1, c='r', label='Class 1')
    plt.scatter(x_cls2, y_cls2, c='b', label='Class 2')
    plt.scatter(new_spot1[0],new_spot1[1],c='g',label='New Spot1')
    plt.scatter(new_spot2[0],new_spot2[1],c='y',label='New Spot2')
    plt.title('Scatter Plot of Two Classes')
    plt.legend()
    plt.show()


