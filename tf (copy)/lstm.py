from utils import *
from lstm_cell import *
import numpy as np

class LSTM(object):

    def __init__(self, D, H):

        self.N, self.T, self.H, self.D = N, T, H, D

        self.Wx = tf.random_normal([D, 4 * H], dtype=tf.float64)
        self.Wx /= np.sqrt(D)
        self.Wx = tf.Variable(self.Wx, name='Wx')
        self.Wh = tf.random_normal([H, 4 * H], dtype=tf.float64)
        self.Wh /= np.sqrt(H)
        self.Wh = tf.Variable(self.Wh, name='Wh')
        self.b = tf.zeros([4 * H], dtype=tf.float64)
        self.b = tf.Variable(self.b, name='b')
        self.h = tf.zeros([N, T+1, H], dtype=tf.float64)
        self.h = tf.Variable(self.h, name="h")
        self.c = tf.zeros([N, T+1, H], dtype=tf.float64)
        self.c = tf.Variable(self.c, name="c")
        self.dx = tf.zeros([self.N, self.T, self.D], dtype=tf.float64)
        self.dx = tf.Variable(self.dx, name="dx")
        self.dprev_h = tf.zeros([self.N, self.H], dtype=tf.float64)
        self.dprev_h = tf.Variable(self.dprev_h, name="dprev_h")
        self.dprev_c = tf.zeros([self.N, self.H], dtype=tf.float64)
        self.dprev_c = tf.Variable(self.dprev_c, name="dprev_c")
        self.dWx = tf.zeros_like(self.Wx, dtype=tf.float64)
        self.dWx = tf.Variable(self.dWx, name="dWx")
        self.dWh = tf.zeros_like(self.Wh, dtype=tf.float64)
        self.dWh = tf.Variable(self.dWh, name="dWh")
        self.db = tf.zeros_like(self.b, dtype=tf.float64)
        self.db = tf.Variable(self.db, name="db")

    def forward(self, x, h0):
        self.x = x

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

        self.h.assign(tf.transpose(h, perm=[1, 0, 2]))
        self.c.assign(tf.transpose(c, perm=[1, 0, 2]))

        return self.h[:, :-1, :]

    def backward(self, dh):
        dx = []

        for j in range(self.T-1, -1, -1):
            dx_j, dprev_h, dprev_c, dWx_j, dWh_j, db_j = self.lstm_cells[j].backward(dh[:, j, :] + dprev_h, dprev_c)
            dx.append(dx_j)
            dWx.assign_add(dWx_j)
            dWh.assign_add(dWh_j)
            db.assign_add(db_j)

        self.dx.assign(tf.transpose(tf.pack(dx), perm=[1, 0, 2]))
        return dx, dprev_h, dWx, dWh, db


