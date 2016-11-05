import numpy as np
from utils import *
from lstm_cell import *

class LSTM(object):

    def __init__(self, D, H):
        self.Wx = np.random.randn(D, 4 * H)
        self.Wx /= np.sqrt(D)
        self.Wh = np.random.randn(H, 4 * H)
        self.Wh /= np.sqrt(H)
        self.b = np.zeros(4 * H)
        self.H = H

    def getParams(self):
        return (self.Wx, self.Wh, self.b)

    def setParams(self, Wx, Wh, b):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b

    def forward(self, x, h0):

        self.x = x
        self.N, self.T, _ = x.shape


        self.lstm_cells = [LSTMCell() for i in range(self.T)]

        self.h = np.zeros((self.N, self.T + 1, self.H))
        self.c = np.zeros((self.N, self.T + 1, self.H))

        self.h[:, -1, :] = h0

        for j in range(self.T):
            self.h[:, j, :], self.c[:, j, :] = self.lstm_cells[j].forward(x[:, j, :], self.h[:, j-1, :], self.c[:, j-1, :], self.Wx, self.Wh, self.b)

        return self.h[:, :-1, :]

    def backward(self, dh):
        dx = np.zeros_like(self.x)
        dprev_h = np.zeros((self.N, self.H))
        dprev_c = np.zeros((self.N, self.H))
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)

        for j in range(self.T-1, -1, -1):
            dx[:, j, :], dprev_h, dprev_c, dWx_j, dWh_j, db_j = self.lstm_cells[j].backward(dh[:, j, :] + dprev_h, dprev_c)
            dWx += dWx_j
            dWh += dWh_j
            db += db_j

        return dx, dprev_h, dWx, dWh, db


