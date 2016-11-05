import tensorflow as tf
import numpy as np

class WordEmbedding(object):

    def __init__(self, V, D):
        
        self.N, self.T, self.V, self.D  = N, V, D, T

        self.W = tf.random_normal([V, D], dtype=tf.float64)
        self.W /= 100
        self.W = tf.Variable(self.W, name='W_embed')
        self.dW = np.zeros((self.V, self.D), dtype=np.float64)

    def forward(self, x):
        self.x = x
        x_flat = tf.reshape(x, [-1])
        out = tf.zeros([self.N * self.T, self.D], dtype=tf.float64)
        out = tf.gather(self.W, x_flat)
        out = tf.reshape(out, [self.N, self.T, self.D])

        return out

    def backward(self, dout):
        np.add.at(dW, self.x.eval().ravel(), dout.eval().reshape(-1, self.D))
        dW = tf.convert_to_tensor(dW)
        
        return dW