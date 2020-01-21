from KNN import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
data , target = data.data , data.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

knn = KNN(x=x_train, y=y_train, k=3)

knn.prediction(tests=x_test, ys=y_test)