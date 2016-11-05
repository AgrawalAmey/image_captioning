# As usual, a bit of setup

import time, os, json
import numpy as np
import matplotlib.pyplot as plt


from np.net import *
from np.utils import *
from np.lstm import *
from np.lstm_cell import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data = load_coco_data(max_train=50)

net = Net(data,
           num_epochs=50,
           batch_size=25,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.995,
           verbose=True, print_every=10,
           word_to_idx=data['word_to_idx'],
           input_dim=data['train_features'].shape[1],
           hidden_dim=512,
           wordvec_dim=256
         )

net.train()

# Plot the training losses
plt.plot(net.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()



for split in ['train', 'val']:
  minibatch = sample_coco_minibatch(data, split=split, batch_size=2)
  gt_captions, features, urls = minibatch
  gt_captions = decode_captions(gt_captions, data['idx_to_word'])

  sample_captions = net.sample(features)
  sample_captions = decode_captions(sample_captions, data['idx_to_word'])

  for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
    plt.imshow(image_from_url(url))
    plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
    plt.axis('off')
    plt.show()