import numpy as np
import tensorflow as tf

class TemporalAffine(object):

    def __init__(self, D, M):
        self.W = tf.random_normal([D, M], dtype=tf.float64)
        self.W /= np.sqrt(D)
        self.b = tf.zeros(M, dtype=tf.float64)
        self.D, self.M = D, M

    def getParams(self):
        return (self.W, self.b)

    def setParams(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        self.N = x.get_shape()[0].value
        self.T = x.get_shape()[1].value

        tmp = tf.matmul(tf.reshape(x, [self.N * self.T, self.D]), self.W)
        out = tf.reshape(tmp, [self.N, self.T, self.M]) + self.b

        return out


    def backward(self, dout):
        dx = tf.matmul(tf.reshape(dout, [self.N * self.T, self.M]), self.W, transpose_b = True)
        dx = tf.reshape(dx, [self.N, self.T, self.D])
        dW = tf.matmul(tf.reshape(self.x, [self.N * self.T, self.D]), tf.reshape(dout, [self.N * self.T, self.M]), transpose_a = True)
        db = tf.reduce_sum(dout, reduction_indices = [0, 1])
        return dx, dW, db