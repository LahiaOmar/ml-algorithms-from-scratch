from Node import Node
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, x, y):
        self._nb_features = x.shape[1]
        self._nb_classes = len(set(y))
        self._all_class = [cur for cur in set(y)]
        self._normalize_classes(y)
        self.tree = self._tree(x, y, 0)

    def _most_common(self, y):
        count = Counter(y)
        freq_classes = [0]*self._nb_classes
        for ind in range(len(freq_classes)):
            freq_classes[ ind ] = count[ self._all_class[ ind]]
        most_common = count.most_common(1)
        pred_class = most_common[0][0]
        return freq_classes, pred_class

    def _tree(self, x, y, depth):
        _, pred_class = self._most_common(y)
        node = Node(label=pred_class)
        if depth < self.max_depth:
            best_index, best_threshold, best_gini = self._split(x,y)
            if best_index is not None and best_threshold is not None:
                left_indices = x[:, best_index ] <= best_threshold
                left_data_x , left_data_y= x[left_indices] , y[left_indices]
                right_data_x, right_data_y = x[~left_indices], y[~left_indices]
                node.threshold = best_threshold
                node.left = self._tree(left_data_x, left_data_y, depth + 1)
                node.right = self._tree(right_data_x, right_data_y, depth + 1)
                node.index = best_index
                node.gini_score = best_gini
        return node

    def predict(self, sample):
        root = self.tree
        while root.left is not None:
            if sample[ root.index ] <= root.threshold:
                root = root.left
            else :
                root = root.right
        return root.label

    def _split(self, x, y):
        nb_current_simples = y.size
        if nb_current_simples <= 1:
            return None, None, None
        freq_classes, _ = self._most_common(y)
        best_gini = 1.0 - sum(   (cur_freq / nb_current_simples  ) ** 2 for cur_freq in freq_classes )
        best_index, best_threshold = None, None
        for i in range(self._nb_features):
            curent_feature = x[:, i]
            sorted_threshold , sorted_class = zip( *sorted(zip( curent_feature, y)) )
            freq_left = [0]*(self._nb_classes)
            freq_right = freq_classes.copy()
            for j in range(1, nb_current_simples):
                curent_class = sorted_class[j-1]
                freq_left[curent_class] += 1
                freq_right[curent_class] -= 1
                gini_left = 1.0 - sum( (fr/(j))**2 for fr in freq_left )
                gini_right = 1.0 - sum((fr / (nb_current_simples - j)) ** 2 for fr in freq_right)
                gini_result = ( (j)*gini_left + (nb_current_simples - j)*gini_right ) / nb_current_simples
                if sorted_threshold[j] == sorted_threshold[j-1] :
                    continue
                if gini_result < best_gini :
                    best_gini = gini_result
                    best_index = i
                    best_threshold = (sorted_threshold[j] + sorted_threshold[j-1]) / 2
        return best_index, best_threshold, best_gini

    def _normalize_classes(self, y):
        unique_classes = np.unique(y)
        if unique_classes[0] != 0:
            for cls in unique_classes:
                y[ y == cls ] = cls-1