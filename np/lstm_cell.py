import numpy as np
from utils import *

class LSTMCell(object):

    def __init__(self):
        pass

    def forward(self, x, prev_h, prev_c, Wx, Wh, b):

        self.x, self.Wx, self.Wh, self.prev_h, self.prev_c = x, Wx, Wh, prev_h, prev_c

        H = prev_h.shape[1]

        # a = prev_h.Wh + x.Wx + b
        a = x.dot(Wx) + prev_h.dot(Wh) + b.reshape((1, -1))

        # Calculating gate values
        self.i = sigmoid(a[:, :H])
        self.f = sigmoid(a[:, H:2 * H])
        self.o = sigmoid(a[:, 2 * H:3 * H])
        self.g = np.tanh(a[:, 3 * H:4 * H])

        self.next_c = (self.f * prev_c) + (self.i * self.g)
        self.next_h = self.o * np.tanh(self.next_c)

        return self.next_h, self.next_c

    def backward(self, dnext_h, dnext_c):

        temp = dnext_c + dnext_h * self.o * (1 - np.tanh(self.next_c) * np.tanh(self.next_c))

        di = temp * self.g
        df = temp * self.prev_c
        dg = temp * self.i
        do = dnext_h * np.tanh(self.next_c)

        dprev_c = temp * self.f

        di_a = di * self.i * (1 - self.i)
        df_a = df * self.f * (1 - self.f)
        do_a = do * self.o * (1 - self.o)
        dg_a = dg * (1 - self.g * self.g)

        da = np.hstack((di_a, df_a, do_a, dg_a))
        db = np.sum(da, axis=0)
        dx = da.dot(self.Wx.T)
        dprev_h = da.dot(self.Wh.T)
        dWx = self.x.T.dot(da)
        dWh = self.prev_h.T.dot(da)

        return dx, dprev_h, dprev_c, dWx, dWh, db