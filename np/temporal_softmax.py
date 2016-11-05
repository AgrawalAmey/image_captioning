import numpy as np

class TemporalSoftmax(object):

    def __init__(self):
        pass

    def forward(self, x, y, mask):

		self.N, self.T, self.V = x.shape
		
		x_flat = x.reshape(self.N * self.T, self.V)
		y_flat = y.reshape(self.N * self.T)
		mask_flat = mask.reshape(self.N * self.T)
		
		probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		loss = -np.sum(mask_flat * np.log(probs[np.arange(self.N * self.T), y_flat])) / self.N

		self.y_flat, self.mask_flat, self.probs = y_flat, mask_flat, probs

		return loss

    def backward(self):
		
		dx_flat = self.probs.copy()
		dx_flat[np.arange(self.N * self.T), self.y_flat] -= 1
		dx_flat /= self.N
		dx_flat *= self.mask_flat[:, None]
		
		dx = dx_flat.reshape(self.N, self.T, self.V)
		
		return dx
