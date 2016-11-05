import os, json
import numpy as np
import h5py
import urllib2, os, tempfile
from scipy.misc import imread
import tensorflow as tf

def adam(x, dx, config=None):
	if config is None: config = {}
	config.setdefault('learning_rate', 1e-3)
	config.setdefault('beta1', 0.9)
	config.setdefault('beta2', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('m', tf.zeros_like(x, dtype=tf.float64))
	config.setdefault('v', tf.zeros_like(x, dtype=tf.float64))
	config.setdefault('t', 0)
	
	next_x = None
	beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
	t, m, v = config['t'], config['m'], config['v']
	m = beta1 * m + (1 - beta1) * dx
	v = beta2 * v + (1 - beta2) * (dx * dx)
	t += 1
	alpha = config['learning_rate'] * tf.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
	alpha = tf.cast(alpha, tf.float64)
	next_x = tf.sub(x, alpha * (m / (tf.sqrt(v) + eps)))
	
	config['t'] = t
	config['m'] = m
	config['v'] = v
	
	
	return next_x, config

def load_coco_data(base_dir='np/datasets',
									 max_train=None,
									 pca_features=True):
	data = {}
	caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
	with h5py.File(caption_file, 'r') as f:
		for k, v in f.iteritems():
			data[k] = np.asarray(v)

	if pca_features:
		train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
	else:
		train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
	with h5py.File(train_feat_file, 'r') as f:
		data['train_features'] = np.asarray(f['features'])

	if pca_features:
		val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
	else:
		val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
	with h5py.File(val_feat_file, 'r') as f:
		data['val_features'] = np.asarray(f['features'])

	dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
	with open(dict_file, 'r') as f:
		dict_data = json.load(f)
		for k, v in dict_data.iteritems():
			data[k] = v

	train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
	with open(train_url_file, 'r') as f:
		train_urls = np.asarray([line.strip() for line in f])
	data['train_urls'] = train_urls

	val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
	with open(val_url_file, 'r') as f:
		val_urls = np.asarray([line.strip() for line in f])
	data['val_urls'] = val_urls

	# Maybe subsample the training data
	if max_train is not None:
		num_train = data['train_captions'].shape[0]
		mask = np.random.randint(num_train, size=max_train)
		data['train_captions'] = data['train_captions'][mask]
		data['train_image_idxs'] = data['train_image_idxs'][mask]

	return data


def decode_captions(captions, idx_to_word):
	singleton = False
	if captions.ndim == 1:
		singleton = True
		captions = captions[None]
	decoded = []
	N, T = captions.shape
	for i in xrange(N):
		words = []
		for t in xrange(T):
			word = idx_to_word[captions[i, t]]
			if word != '<NULL>':
				words.append(word)
			if word == '<END>':
				break
		decoded.append(' '.join(words))
	if singleton:
		decoded = decoded[0]
	return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
	split_size = data['%s_captions' % split].shape[0]
	mask = np.random.choice(split_size, batch_size)
	captions = data['%s_captions' % split][mask]
	image_idxs = data['%s_image_idxs' % split][mask]
	image_features = data['%s_features' % split][image_idxs]
	urls = data['%s_urls' % split][image_idxs]
	return captions, image_features, urls

def image_from_url(url):
	try:
		f = urllib2.urlopen(url)
		_, fname = tempfile.mkstemp()
		with open(fname, 'wb') as ff:
			ff.write(f.read())
		img = imread(fname)
		os.remove(fname)
		return img
	except urllib2.URLError as e:
		print 'URL Error: ', e.reason, url
	except urllib2.HTTPError as e:
		print 'HTTP Error: ', e.code, url

