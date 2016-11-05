import numpy as np

class TemporalAffine(object):

    def __init__(self, D, M):
        self.W = np.random.randn(D, M)
        self.W /= np.sqrt(D)
        self.b = np.zeros(M)
        self.D, self.M = D, M

    def getParams(self):
        return (self.W, self.b)

    def setParams(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
		self.x = x
		self.N, self.T, _ = x.shape		

		out = x.reshape(self.N * self.T, self.D).dot(self.W).reshape(self.N, self.T, self.M) + self.b
		
		return out

    def backward(self, dout):
		dx = dout.reshape(self.N * self.T, self.M).dot(self.W.T).reshape(self.N, self.T, self.D)
		dW = dout.reshape(self.N * self.T, self.M).T.dot(self.x.reshape(self.N * self.T, self.D)).T
		db = dout.sum(axis=(0, 1))
	
		return dx, dW, db