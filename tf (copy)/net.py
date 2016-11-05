import numpy as np

from utils import *
from lstm import *
from lstm_cell import *
from word_embedding import *
from temporal_affine import *
from temporal_softmax import *

class Net(object):

	def __init__(self, data, **kwargs):
		
		self.data = data
		self.H = kwargs.pop('hidden_dim', 128)
		self.D = kwargs.pop('wordvec_dim', 128)
		self.input_dim = kwargs.pop('input_dim', 128)
		self.optim_config = kwargs.pop('optim_config', {})
		self.lr_decay = kwargs.pop('lr_decay', 1.0)
		self.batch_size = kwargs.pop('batch_size', 100)
		self.num_epochs = kwargs.pop('num_epochs', 10)
		self.print_every = kwargs.pop('print_every', 10)
		self.word_to_idx = kwargs.pop('word_to_idx')
		
		self.idx_to_word = {i: w for w, i in self.word_to_idx.iteritems()}
		
		self.V = len(self.word_to_idx)
	
		self._null = self.word_to_idx['<NULL>']
		self._start = self.word_to_idx['<START>']
		self._end = self.word_to_idx['<END>']

		self.lstm = LSTM(self.D, self.H)
		self.word_embedding = WordEmbedding(self.V, self.D)
		self.temporal_affine_h0 = TemporalAffine(self.input_dim, self.H)
		self.temporal_affine_scores = TemporalAffine(self.H, self.V)
		self.temporal_softmax = TemporalSoftmax()
		
		# Unpack keyword arguments

		self.params = {}
		self.params_getter()

		self._reset()

	def params_getter(self):
		self.params['W_embed'] = self.word_embedding.getParams()
		self.params['W_proj'], self.params['b_proj'] = self.temporal_affine_h0.getParams()
		self.params['Wx'], self.params['Wh'], self.params['b'] = self.lstm.getParams()
		self.params['W_vocab'], self.params['b_vocab'] = self.temporal_affine_scores.getParams()

	def params_setter(self):
		self.word_embedding.setParams(self.params['W_embed'])
		self.temporal_affine_h0.setParams(self.params['W_proj'], self.params['b_proj'])
		self.lstm.setParams(self.params['Wx'], self.params['Wh'], self.params['b'])
		self.temporal_affine_scores.setParams(self.params['W_vocab'], self.params['b_vocab'])	

	def loss(self, features, captions):
		loss, grads = 0.0, {}
		N, T = captions.shape
		H = self.H
		D = features.shape[1]

		features = tf.convert_to_tensor(features, dtype=tf.float64)

		captions_in = tf.concat(1, [tf.fill([N, 1], self._start), tf.convert_to_tensor(captions)[:, :-1]])
		captions_out = tf.concat(1, [tf.convert_to_tensor(captions)[:, 1:], tf.fill([N, 1], self._end)])
		
		# You'll need this 
		mask = tf.not_equal(captions_out, self._null)
	
		# Affine transform to get h0 
		temp = self.temporal_affine_h0.forward(tf.reshape(features, [N, 1, D]))
		h0 = tf.reshape(temp, [N, H])

		# word vectors
		x = self.word_embedding.forward(captions_in)

		h = self.lstm.forward(x, h0)

		# Use affine transform to find scores
		scores = self.temporal_affine_scores.forward(h) 

		# Compute loss
		loss = self.temporal_softmax.forward(scores, captions_out, mask)

		### Backward Pass
		dy = self.temporal_softmax.backward()

		dh, dW_vocab, db_vocab = self.temporal_affine_scores.backward(dy)

		dx, dh0, dWx, dWh, db = self.lstm.backward(dh)

		dW_embed = self.word_embedding.backward(dx)

		dh0 = tf.reshape(dh0, [N, 1, H])

		_, dW_proj, db_proj = self.temporal_affine_h0.backward(dh0)	

		grads['W_embed'] = dW_embed
		grads['W_proj'] = dW_proj
		grads['b_proj'] = db_proj
		grads['Wx'] = dWx
		grads['Wh'] = dWh
		grads['b'] = db
		grads['W_vocab'] = dW_vocab
		grads['b_vocab'] = db_vocab
		
		return loss, grads

	def _reset(self):
		"""
		Set up some book-keeping variables for optimization. Don't call this
		manually.
		"""
		# Set up some variables for book-keeping
		self.epoch = 0
		self.best_val_acc = 0
		self.best_params = {}
		self.loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []

		# Make a deep copy of the optim_config for each parameter
		self.optim_configs = {}
		for p in self.params:
			d = {k: v for k, v in self.optim_config.iteritems()}
			self.optim_configs[p] = d

	def _step(self):
		# Make a minibatch of training data
		minibatch = sample_coco_minibatch(self.data,
									batch_size=self.batch_size,
									split='train')
		captions, features, urls = minibatch

		# Compute loss and gradient
		loss, grads = self.loss(features, captions)
		self.loss_history.append(loss.eval())

		# Perform a parameter update
		for p, w in self.params.iteritems():
			dw = grads[p]
			config = self.optim_configs[p]
			next_w, next_config = adam(w, dw, config)
			self.params[p] = next_w
			self.optim_configs[p] = next_config
		self.params_setter()

	def train(self):
		num_train = self.data['train_captions'].shape[0]
		iterations_per_epoch = max(num_train / self.batch_size, 1)
		num_iterations = self.num_epochs * iterations_per_epoch

		for t in xrange(num_iterations):
			self._step()

			# Maybe print training loss
			if t % self.print_every == 0:
				print '(Iteration %d / %d) loss: %f' % (
							 t + 1, num_iterations, self.loss_history[-1])

			# At the end of every epoch, increment the epoch counter and decay the
			# learning rate.
			epoch_end = (t + 1) % iterations_per_epoch == 0
			if epoch_end:
				self.epoch += 1
				for k in self.optim_configs:
					self.optim_configs[k]['learning_rate'] *= self.lr_decay

	def sample(self, features, max_length=30):

		N = tf.shape(features)[0]

		captions = self._null * np.ones((N, max_length+1))
	
		# Unpack parameters
		H, V, D = self.H, self.V, self.D

		# Initializing h, c and x
		h = self.temporal_affine_h0.forward(tf.reshape(features, [N, 1, -1]))
		h = tf.reshape(h, [N, H])
		c = tf.zeros([N, H], dtype=tf.float64)
		captions[:, 0] = self._start

		lstm_cell = LSTMCell()

		for i in range(max_length):
			# Word embedding
			x = self.word_embedding.forward(tf.reshape(captions[:, i-1], [N, 1]))
			x = tf.reshape(x, [N, D])
			# RNN forward
			h, c = lstm_cell.forward(x, h, c, self.params['Wx'], self.params['Wh'], self.params['b'])
			# Scores
			scores = self.temporal_affine_scores.forward(tf.reshape(h, [N, 1, H])) 
			scores = tf.reshape(scores, [N, V])
			# Add next word
			captions[:, i] = tf.argmax(scores, [1]).eval()

		return captions[:, 1:]


