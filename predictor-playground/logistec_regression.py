import numpy as np

class LogisticRegression:
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def sigmoid(self,h):
        return (1/(1+np.exp(-h)))