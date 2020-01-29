from kmeans import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
data, target = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
km = KMeans(data=x_train, n_cluster=3, epsilon=0.01)
km.prediction(x_test=x_test)