import numpy as np

class WordEmbedding(object):

    def __init__(self, V, D):
        self.W = np.random.randn(V, D)
        self.W /= 100
        self.V, self.D  = V, D

    def getParams(self):
        return self.W

    def setParams(self, W):
        self.W = W

    def forward(self, x):
        self.x = x
        N, T = x.shape

        x_flat = list(self.x.ravel())
        out = np.zeros((N * T, self.D))
        out = self.W[x_flat, :]
        out = out.reshape((N, T, self.D))

        return out

    def backward(self, dout):
        dW = np.zeros((self.V, self.D))

        np.add.at(dW, self.x.ravel(), dout.reshape(-1, self.D))

        return dW