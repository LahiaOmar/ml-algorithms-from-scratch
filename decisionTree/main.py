from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import load_iris

dataset = load_iris()
X, Y = dataset.data , dataset.target
clf = DecisionTree(10)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
clf.fit(x_train, y_train)
counte = 0
for simple , p_simple in zip(x_test, y_test):
    pred = clf.predict(simple)
    print(pred, p_simple)
    if pred == p_simple :
        counte += 1
print("counter : {} all {}".format(counte, y_test.shape))