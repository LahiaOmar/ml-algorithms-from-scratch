import numpy as np
import random
from scipy.spatial import distance


class Clusters:
    def __init__(self, nb_clusters, size_cluster):
        self._nb_cluster = nb_clusters
        self.clts = [[] for _ in range(self._nb_cluster)]
        self.diff = []
        self._new_clusters = [[] for _ in range(self._nb_cluster)]
        self.reset_centroids()

    def add(self, arr, to):
        self._new_clusters[to].append(arr)

    def means_cluster(self):
        cl = np.array(self._new_clusters)
        means = [ np.mean(clt, axis=0) for clt in self._new_clusters]
        # means = np.mean(cl, axis=0)
        self.diff = [ distance.euclidean(c, cc) for c, cc in zip(self.clts, means) ]
        self.clts = means

    def reset_centroids(self):
        self._new_clusters= [[] for _ in range(self._nb_cluster)]

    def set_centroid(self, index, data):
        self._new_clusters[index].append(data)
        self.clts[index].append(data)

    def get_cluster(self, index):
        return self.clts[index]

    def get_difference(self):
        return np.mean(self.diff)

    def get_all_clusters(self):
        return self.clts
