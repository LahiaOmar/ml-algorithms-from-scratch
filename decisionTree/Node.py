class Node:
    def __init__(self, label):
        self.label = label
        self.left = None
        self.right = None
        self.gini_score = None
        self.threshold = None
        self.index = None
        print("label of Node : {}".format(self.label))

    def __repr__(self):
        return "label : {} threshold {} gini = {} \n left {} right {}".format(self.label, self.threshold, self.gini_score,self.left, self.right)
