import tensorflow as tf

class TemporalSoftmax(object):

    def __init__(self, N, T, V):
        self.N, self.T, self.V = N, T, V

    def forward(self, x, y, mask):
		
		x_flat = tf.reshape(x, [self.N * self.T, self.V])
		y_flat = tf.reshape(y, [self.N * self.T])
		mask_flat = tf.cast(tf.reshape(mask, [self.N * self.T]), tf.float64)
		
		probs = tf.exp(x_flat - tf.reduce_max(x_flat, reduction_indices=[1], keep_dims=True))
		probs /= tf.reduce_sum(probs, reduction_indices=[1], keep_dims=True)
		coords = tf.transpose(tf.pack([tf.range(self.N * self.T), y_flat]))
		loss = -tf.reduce_sum(mask_flat * tf.log(tf.gather_nd(probs, coords))) / self.N

		self.y_flat, self.mask_flat, self.probs = y_flat, mask_flat, probs

		return loss

    def backward(self):
		
		dx_flat = self.probs

		coords = tf.transpose(tf.pack([tf.range(self.N * self.T), self.y_flat]))
		binary_mask = tf.sparse_to_dense(coords, dx_flat.get_shape(), 1)
		# convert 1/0 to True/False
		binary_mask = tf.cast(binary_mask, tf.bool)
		decremented = dx_flat - 1
		# make new x out of old values or decresed, depending on mask 
		dx_flat = tf.select(binary_mask, decremented, dx_flat)
		dx_flat /= self.N
		dx_flat *= self.mask_flat[:, None]
		
		dx = tf.reshape(dx_flat, [self.N, self.T, self.V])
		
		return dx
