import tensorflow as tf
import numpy as np

class WordEmbedding(object):

    def __init__(self, V, D):
        self.W = tf.random_normal([V, D], dtype=tf.float64)
        self.W /= 100
        self.V, self.D  = V, D

    def getParams(self):
        return self.W

    def setParams(self, W):
        self.W = W

    def forward(self, x):
        self.x = x
        self.N = x.get_shape()[0].value 
        self.T = x.get_shape()[1].value

        x_flat = tf.reshape(x, [-1])
        out = tf.zeros([self.N * self.T, self.D], dtype=tf.float64)
        out = tf.gather(self.W, x_flat)
        out = tf.reshape(out, [self.N, self.T, self.D])

        return out

    def backward(self, dout):
        dW = np.zeros((self.V, self.D), dtype=np.float64)
        np.add.at(dW, self.x.eval().ravel(), dout.eval().reshape(-1, self.D))
        dW = tf.convert_to_tensor(dW)
        return dW