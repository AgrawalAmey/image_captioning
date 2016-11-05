import numpy as np
import tensorflow as tf

class TemporalAffine(object):

    def __init__(self, N, T, D, M, name):

        self.N, self.T, self.D, self.M = N, T, D, M

        self.W = tf.random_normal([D, M], dtype=tf.float64)
        self.W /= np.sqrt(D)
        self.W = tf.Variable(self.W, name=('W_'+name))
        self.b = tf.zeros(M, dtype=tf.float64)
        self.b = tf.Variable(self.b, name=('b_'+name))
        self.dx = tf.Variable(tf.zeros([N, T, M], dtype=tf.float64), name=('dx_'+name))
        self.dW = tf.Variable(tf.zeros([D, M], dtype=tf.float64), name=('dW_'+name))
        self.db = tf.Variable(tf.zeros([M], dtype=tf.float64), name=('db_'+name))

    def forward(self, x):
        self.x = x

        tmp = tf.matmul(tf.reshape(x, [self.N * self.T, self.D]), self.W)
        out = tf.reshape(tmp, [self.N, self.T, self.M]) + self.b

        return out

    def backward(self, dout):
        dx.assign(tf.matmul(tf.reshape(dout, [self.N * self.T, self.M]), self.W, transpose_b = True))
        dx.assign(tf.reshape(dx, [self.N, self.T, self.D]))
        dW.assign(tf.matmul(tf.reshape(self.x, [self.N * self.T, self.D]), tf.reshape(dout, [self.N * self.T, self.M]), transpose_a = True))
        db.assign(tf.reduce_sum(dout, reduction_indices = [0, 1]))
        
        return dx, dW, db