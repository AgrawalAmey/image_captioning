from utils import *
from lstm_cell import *
import numpy as np

class LSTM(object):

    def __init__(self, D, H):
        self.Wx = tf.random_normal([D, 4 * H], dtype=tf.float64)
        self.Wx /= np.sqrt(D)
        self.Wh = tf.random_normal([H, 4 * H], dtype=tf.float64)
        self.Wh /= np.sqrt(H)
        self.b = tf.zeros([4 * H], dtype=tf.float64)
        self.H = H

    def getParams(self):
        return (self.Wx, self.Wh, self.b)

    def setParams(self, Wx, Wh, b):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b

    def forward(self, x, h0):

        self.x = x
        self.N = x.get_shape()[0].value
        self.T = x.get_shape()[1].value

        self.lstm_cells = [LSTMCell() for i in range(self.T)]

        h, c = [], []
        h.append(h0)
        c.append(tf.zeros([self.N, self.H], dtype=tf.float64))

        for j in range(self.T):
            h_j, c_j = self.lstm_cells[j].forward(x[:, j, :], h[j-1], c[j-1], self.Wx, self.Wh, self.b)
            h.append(h_j)
            c.append(c_j)

        h = tf.pack(h)
        c = tf.pack(c)

        self.h = tf.transpose(h, perm=[1, 0, 2])
        self.c = tf.transpose(c, perm=[1, 0, 2])

        return self.h[:, :-1, :]

    def backward(self, dh):
        dx = []
        dprev_h = tf.zeros([self.N, self.H], dtype=tf.float64)
        dprev_c = tf.zeros([self.N, self.H], dtype=tf.float64)
        dWx = tf.zeros_like(self.Wx, dtype=tf.float64)
        dWh = tf.zeros_like(self.Wh, dtype=tf.float64)
        db = tf.zeros_like(self.b, dtype=tf.float64)

        for j in range(self.T-1, -1, -1):
            dx_j, dprev_h, dprev_c, dWx_j, dWh_j, db_j = self.lstm_cells[j].backward(dh[:, j, :] + dprev_h, dprev_c)
            dx.append(dx_j)
            dWx += dWx_j
            dWh += dWh_j
            db += db_j

        dx = tf.pack(dx)
        dx = tf.transpose(dx, perm=[1, 0, 2])
        return dx, dprev_h, dWx, dWh, db


