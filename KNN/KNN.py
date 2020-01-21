import numpy as np
from scipy.spatial import distance
from collections import Counter
class KNN:
    def __init__(self, x, y , k):
        self.k = k
        self.data = x
        self.nb_rows = x.shape[0]
        self.y = y

    def distance(self, x, y):
        return distance.euclidean(x,y)

    def _get_min(self, old, new):
        if old[0] < new[0] :
            return old
        return new

    def _get_most(self, arr):
        return Counter(arr).most_common(1)[0][0]

    def get_distance(self, test, index):
        di = self.distance(test, self.data[index])
        return di, self.y[index]

    def _print_accuracy(self, y, p):
        count = 0
        for i in range(y.shape[0]):
            if y[i] == p[i]:
                count += 1
        print("accuracy : {}".format(count / y.shape[0]))

    def prediction(self, tests, ys):
        preds = []
        for test in tests:
            all = []
            for index in range(self.nb_rows):
                all.append(self.get_distance(test, index))
            all.sort(key = lambda element : element[0])
            clss = [cur[1] for cur in all[:self.k]]
            most = self._get_most(clss)
            preds.append(most)
        self._print_accuracy(ys, preds)