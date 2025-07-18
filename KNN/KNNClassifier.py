import numpy as np
from math import sqrt
from collections import Counter
from sklearn import datasets

class KNNClassifier:

    def __init__(self,k):
        
        assert k>=1,"k must be valid"
        self.k=k
        self.cls1=None
        self.cls2=None

    def setClass(self,cls1,cls2):
        
        assert self.k<=cls1.shape[0],"k must be smaller than class1"
        assert self.k<=cls2.shape[0],"k must be smaller than class2"

        self.cls1=cls1
        self.cls2=cls2

        return self

    def predict(self, new_spot):
        
        distances=self._calculateDistance(new_spot)
        
        vote_result=self._vote(distances)
        return vote_result
    
    def _calculateDistance(self,new_spot):

        distances=[]
        #Calculate the distance between new_spot and cls1
        for i in range(len(self.cls1)):
            distance=sqrt((new_spot[0]-self.cls1[i,0])**2+(new_spot[1]-self.cls1[i,1])**2)
            distances.append(distance)
        
        #Calculate the distance between new_spot and cls2
        for i in range(len(self.cls2)):
            distance=sqrt((new_spot[0]-self.cls2[i,0])**2+(new_spot[1]-self.cls2[i,1])**2)
            distances.append(distance)

        return distances

    def _vote(self,distances):

         #Find the nearest k points
        #Sort distances asc, and get the order number
        nearest=np.argsort(distances)
        #Get the first k nearest points
        nearest_k=nearest[:self.k]
        nearest_cls=[]
        for i in range(len(nearest_k)):
            if nearest_k[i]<20:
                nearest_cls.append(1)
            else:
                nearest_cls.append(2)
        
        votes=Counter(nearest_cls)
        vote_result=votes.most_common(1)[0][0]
        return vote_result


class KNNClassifierCommon:

    def __init__(self,k):
        
        assert k>=1,"k must be valid"
        self.k=k
        self.data=None
        self.target=None

    def fit(self,data,target):
        
        assert data.shape[0]==target.shape[0],"the size of data must be equal to the size of target"
        assert self.k<=data.shape[0],"k must be smaller than the number of samples"

        self.data=data
        self.target=target
        return self

    def predict(self,testData):
        
        assert self.data is not None and self.target is not None, "Please fit before predict"
        #assert testData.shape[1]==self.data.shape[1],"the feature number of testData must be equal to data"
        
        newTarget=[self._predict(x) for x in testData]
        return np.array(newTarget)
    
    def _predict(self,x):
       
       #assert x.shape[0]==self.data.shape[1],"the feature number of x must be equal to data"

       distance=[sqrt(np.sum((x_train-x)**2)) for x_train in self.data]
       nearest=np.argsort(distance)
       topK_y=[self.target[i] for i in nearest[:self.k]]
       votes=Counter(topK_y)
       return votes.most_common(1)[0][0]