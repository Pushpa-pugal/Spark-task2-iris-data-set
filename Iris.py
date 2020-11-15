#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#import datasets
df = "E:/MBA/spark foundation/task 2/Iris.csv"
iris= pd.read_csv(df)
x = iris.iloc[:,[1,2,3,4]].values
print(iris.head(5))


#checking dataset for datatypes and missing data
print(iris.dtypes)
print (iris.describe(include='all'))
#Species has 3 unique values i.e., 3 unique clusters will be optimal
print(iris.info())


#finding the optimum number of clusters for k-means classification to check
wcss=[]
K=range(1,11)
for i in K:
    k = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    y_k = k.fit(x)
    wcss.append(k.inertia_)
print(wcss)

#find the elbow point
import kneed
kn = kneed.KneeLocator(K, wcss, curve='convex', direction='decreasing')
print("\nknee=",kn.knee)

#plot sum of squared distances
plt.plot(K, wcss, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances-wcss')
plt.title('Elbow Method to find Optimal k')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.show()

for i in K:
    k = KMeans(n_clusters = 3, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    y_k = k.fit_predict(x)
    wcss.append(k.inertia_)
print(wcss)
#plot clusters
plt.scatter(x[y_k == 0, 0], x[y_k == 0, 1], s = 70, c = 'green', label = 'Iris-versicolour')
plt.scatter(x[y_k == 1, 0], x[y_k == 1, 1], s = 70, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_k == 2, 0], x[y_k == 2, 1], s = 70, c = 'blue', label = 'Iris-virginica')
plt.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:,1], s = 70, c = 'black', label = 'Centroids')
plt.legend()






