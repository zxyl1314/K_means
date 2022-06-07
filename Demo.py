import dm2022exp
from collections import Counter
import numpy as np
import random
import math

class KMeans(object):

    def __init__(self, num_cluster : int, max_iter : int =1000,
                 tol : float=1e-4):
        self.num_cluster = num_cluster
        self.max_iter = max_iter
        self.tol = tol
        pass

    def fit(self, X : np.ndarray) -> np.ndarray:
        Distance = math.inf
        L = [i for i in range(len(X))]
        L = np.random.choice(L, self.num_cluster, replace=False)
        random_Centroids = []
        for i in L:
            random_Centroids.append(X[i]) #计算最开始的质心
        Class, Centroids = self.ReturnClass(X, random_Centroids) # 将每个数据分类,
        while Distance >= self.tol:
            Averages = []
            temp = 0
            for i in set(Class): #重新选取每个类的质心
                Average = self.ReturnAverage(Class, i, X)#某一类别的均值质心
                Averages.append(Average)
                temp += math.sqrt(sum([(Average[j] - Centroids[i][j]) ** 2 for j in range(len(Centroids[0]))]))
            Distance = temp
            if Distance < self.tol:
                return np.array(Class)
            else:
                Class, Centroids = self.ReturnClass(X, Averages)



    def ReturnAverage(self,Class,Idx, Dataset):#返回均值质心
        Average = np.zeros((1, Dataset.shape[1]))[0]
        Count = 0
        for i in range(len(Dataset)):
            if Class[i] == Idx:
                Count += 1
                for j in range(Dataset.shape[1]):
                    Average[j] += Dataset[i][j]
        for i in range(len(Average)):
            Average[i] = Average[i] / Count

        return Average

    def ReturnClass(self, X, Centroids):
        Class_Idx = []
        for Attribute in X:
            Idx = None #应划分的类别下标
            Distance = math.inf #将最初距离设置为无穷大
            for i in range(len(Centroids)): #计算每一个点到质心的距离
                present_distance = 0
                for j in range(X.shape[1]):
                    present_distance += (Attribute[j] - Centroids[i][j]) ** 2
                if present_distance < Distance: #选取最短距离的质心作为该数据的类别
                    Distance = present_distance #记录当前的最短距离
                    Idx = i  #类别下标
            Class_Idx.append(Idx) #将该数据划分到对应的类别下
        return Class_Idx , Centroids

def purity(y_pred, y, n_class):
    pr = 0
    for i in range(n_class):
        idxs = y_pred == i
        cnt  = Counter(y[idxs])
        pr += cnt.most_common()[0][1]
    return pr / len(y)

if __name__ == '__main__':
    X, y = dm2022exp.load_ex4_data()
    dm2022exp.show_exp4_data(X, y)
    print("The X Shape is:", X.shape)
    print("The y Shape is:", y.shape)
    print(X)
    print(y)

    KM = KMeans(8)
    res = KM.fit(X)
    dm2022exp.show_exp4_data(X, res)
    print("-----result------")
    purity_rate = purity(res, y, 8)
    print(res)
    print(purity(res, y, 8))
    #print("the purity_rate is:", purity_rate)
pass