import numpy as np
import random
from scipy.spatial import distance
from Clusters import Clusters


class KMeans:
    def __init__(self, data, n_cluster=2, epsilon=0.1):
        self.data = data
        self._n_cluster = n_cluster
        self._epsilon = epsilon
        self._centroids = np.array([], dtype=np.int32)
        self._nb_example = data.shape[0]
        self._size_example = data.shape[1]

        self._clusters = Clusters(nb_clusters=self._n_cluster, size_cluster=self._size_example)
        self._generate_centroids()
        self._init_centroids()
        self._clustering()

    def _init_centroids(self):
        for i in range(self._n_cluster):
            self._clusters.set_centroid(index=i, data=self.data[self._centroids[i]])

    def _clustering(self):
        diff = 99
        while diff > self._epsilon:
            for d in self.data:
                d_distance = 9999999999999.0
                d_index = - 1
                for cl in range(self._n_cluster):
                    cur_distance = self._distance(x=d, y=self._clusters.get_cluster(index=cl))
                    d_distance, d_index = self._min(d_distance, d_index, cur_distance, cl)
                self._clusters.add(d, to=d_index)
            self._clusters.means_cluster()
            self._clusters.reset_centroids()
            diff = self._clusters.get_difference()

    def _min(self,d_distance, d_index, cur_distance, cl):
        if cur_distance < d_distance:
            return cur_distance, cl
        return d_distance, d_index

    def prediction(self, x_test):
        all_clusters = self._clusters.get_all_clusters()
        for test in x_test:
            nearst_cluster = [ self._distance(test, clt) for clt in all_clusters ]
            ind = np.argmax(nearst_cluster)
            print("for {} the class is {} ".format(test, ind))

    def _distance(self, x, y):
        return distance.euclidean(x,y)

    def _generate_centroids(self):
        random.seed(200)
        while self._centroids.shape[0] < self._n_cluster:
            gen_ind = random.randint(0, self._nb_example)
            if gen_ind not in self._centroids:
                self._centroids = np.append(self._centroids, gen_ind)