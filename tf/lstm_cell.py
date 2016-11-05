import tensorflow as tf
from utils import *

class LSTMCell(object):

    def __init__(self):
        pass

    def forward(self, x, prev_h, prev_c, Wx, Wh, b):
        self.x, self.Wx, self.Wh, self.prev_h, self.prev_c = x, Wx, Wh, prev_h, prev_c

        H = prev_h.get_shape()[1].value

        # a = prev_h.Wh + x.Wx + b)
        a = tf.matmul(x, Wx) + tf.matmul(prev_h, Wh) + b

        # Calculating gate values
        self.i = tf.sigmoid(a[:, :H])
        self.f = tf.sigmoid(a[:, H:2 * H])
        self.o = tf.sigmoid(a[:, 2 * H:3 * H])
        self.g = tf.tanh(a[:, 3 * H:4 * H])

        self.next_c = (self.f * prev_c) + (self.i * self.g)
        self.next_h = self.o * tf.tanh(self.next_c)

        return self.next_h, self.next_c

    def backward(self, dnext_h, dnext_c):

        temp = dnext_c + dnext_h * self.o * (1 - tf.tanh(self.next_c) * tf.tanh(self.next_c))

        di = temp * self.g
        df = temp * self.prev_c
        dg = temp * self.i
        do = dnext_h * tf.tanh(self.next_c)

        dprev_c = temp * self.f

        di_a = di * self.i * (1 - self.i)
        df_a = df * self.f * (1 - self.f)
        do_a = do * self.o * (1 - self.o)
        dg_a = dg * (1 - self.g * self.g)

        da = tf.concat(1, [di_a, df_a, do_a, dg_a])

        db = tf.reduce_sum(da, reduction_indices=[0])
        dx = tf.matmul(da, self.Wx, False, True)
        dprev_h = tf.matmul(da, self.Wh, False, True)
        dWx = tf.matmul(self.x, da, True, False)
        dWh = tf.matmul(self.prev_h, da, True, False)

        return dx, dprev_h, dprev_c, dWx, dWh, db
