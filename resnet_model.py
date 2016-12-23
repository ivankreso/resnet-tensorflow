import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
import os, re

#import losses

FLAGS = tf.app.flags.FLAGS

MODEL_DEPTH = 50
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
#MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
MEAN_BGR = [103.939, 116.779, 123.68]


def evaluate(name, sess, epoch_num, run_ops, dataset, data):
  loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation(
      sess, epoch_num, run_ops, dataset.num_examples())
  if iou > data['best_iou'][0]:
    data['best_iou'] = [iou, epoch_num]

def print_results(data):
  print('Best validation IOU = %.2f (epoch %d)' % tuple(data['best_iou']))

def init_eval_data():
  train_data = {}
  valid_data = {}
  train_data['lr'] = []
  train_data['loss'] = []
  train_data['iou'] = []
  train_data['accuracy'] = []
  train_data['best_iou'] = [0, 0]
  valid_data['best_iou'] = [0, 0]
  valid_data['loss'] = []
  valid_data['iou'] = []
  valid_data['accuracy'] = []
  return train_data, valid_data

def normalize_input(img):
  return img - MEAN_BGR
  #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  #with tf.name_scope('input'), tf.device('/cpu:0'):
  #  red, green, blue = tf.split(3, 3, rgb)
  #  bgr = tf.concat(3, [blue, green, red])
  #  #bgr -= MEAN_BGR
  #  return bgr

def _build(image, is_training):
  #def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
  weight_decay = 1e-4
  global bn_params
  bn_params = {
    # Decay for the moving averages.
    #'decay': 0.999,
    'decay': 0.9,
    'center': True,
    'scale': True,
    # epsilon to prevent 0s in variance.
    #'epsilon': 0.001,
    'epsilon': 1e-5,
    # None to force the updates
    'updates_collections': None,
    'is_training': is_training,
  }
  #init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
  init_func = layers.variance_scaling_initializer()

  def shortcut(l, n_in, n_out, stride):
    if n_in != n_out:
      return layers.convolution2d(l, n_out, kernel_size=1, stride=stride,
                                  activation_fn=None, scope='convshortcut')
      #l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
      #return BatchNorm('bnshortcut', l)
    else:
      return l

  def bottleneck(l, ch_out, stride, preact):
    ch_in = l.get_shape().as_list()[-1]
    if preact == 'both_preact':
      l = tf.nn.relu(l, name='preact-relu')
    bottom_in = l
    with arg_scope([layers.convolution2d],
      stride=1, padding='SAME', activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=init_func,
      weights_regularizer=layers.l2_regularizer(weight_decay)):

      l = layers.convolution2d(l, ch_out, kernel_size=1, stride=stride, scope='conv1')

      l = layers.convolution2d(l, ch_out, kernel_size=3, scope='conv2')
      l = layers.convolution2d(l, ch_out * 4, kernel_size=1, activation_fn=None, scope='conv3')
      return l + shortcut(bottom_in, ch_in, ch_out * 4, stride)

  def layer(l, layername, features, count, stride, first=False):
    with tf.variable_scope(layername):
      with tf.variable_scope('block0'):
        l = bottleneck(l, features, stride, 'no_preact' if first else 'both_preact')
      for i in range(1, count):
        with tf.variable_scope('block{}'.format(i)):
          l = bottleneck(l, features, 1, 'both_preact')
      return l

  cfg = {
      50: ([3,4,6,3]),
      101: ([3,4,23,3]),
      152: ([3,8,36,3])
  }
  defs = cfg[MODEL_DEPTH]
  
  skip_layers = []
  image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])
  l = layers.convolution2d(image, 64, 7, stride=2, padding='VALID',
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
      activation_fn=tf.nn.relu, weights_initializer=init_func,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
  l = layers.max_pool2d(l, 3, stride=2, padding='SAME', scope='pool0')
  l = layer(l, 'group0', 64, defs[0], 1, first=True)
  l = layer(l, 'group1', 128, defs[1], 2)
  l = layer(l, 'group2', 256, defs[2], 2)
  l = layer(l, 'group3', 512, defs[3], 2)
  l = tf.nn.relu(l)
  in_k = l.get_shape().as_list()[-2]
  l = layers.avg_pool2d(l, kernel_size=in_k, scope='global_avg_pool')
  l = layers.flatten(l, scope='flatten')
  logits = layers.fully_connected(l, 1000, activation_fn=None, scope='fc1000')
  return logits

def name_conversion(caffe_layer_name, prefix=''):
  """ Convert a caffe parameter name to a tensorflow parameter name as
      defined in the above model """
  # beginning & end mapping
  NAME_MAP = {
      'bn_conv1/beta': 'conv0/BatchNorm/beta:0',
      'bn_conv1/gamma': 'conv0/BatchNorm/gamma:0',
      'bn_conv1/mean/EMA': 'conv0/BatchNorm/moving_mean:0',
      'bn_conv1/variance/EMA': 'conv0/BatchNorm/moving_variance:0',
      'conv1/W': 'conv0/weights:0', 'conv1/b': 'conv0/biases:0',
      'fc1000/W': 'fc1000/weights:0', 'fc1000/b': 'fc1000/biases:0'}
  if caffe_layer_name in NAME_MAP:
    return prefix + NAME_MAP[caffe_layer_name]

  s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
  if s is None:
    s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
    layer_block_part1 = s.group(3)
    layer_block_part2 = s.group(4)
    assert layer_block_part1 in ['a', 'b']
    layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
  else:
    layer_block = ord(s.group(3)) - ord('a')
  layer_type = s.group(1)
  layer_group = s.group(2)

  layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
  assert layer_branch in [1, 2]
  if layer_branch == 2:
    layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
    layer_id = ord(layer_id) - ord('a') + 1

  TYPE_DICT = {'res':'conv', 'bn':'BatchNorm'}
  name_map = {'/W': '/weights:0', '/b': '/biases:0', '/beta': '/beta:0',
              '/gamma': '/gamma:0', '/mean/EMA': '/moving_mean:0',
              '/variance/EMA': '/moving_variance:0'}

  tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
  #print(tf_name)
  if tf_name in name_map:
    tf_name = name_map[tf_name]
  #print(layer_type)
  #if layer_type != 'bn':
  if layer_type == 'res':
    layer_type = TYPE_DICT[layer_type] + (str(layer_id)
        if layer_branch == 2 else 'shortcut')
  elif layer_branch == 2:
    layer_type = 'conv' + str(layer_id) + '/' + TYPE_DICT[layer_type]
  elif layer_branch == 1:
    layer_type = 'convshortcut/' + TYPE_DICT[layer_type]
  tf_name = 'group{}/block{}/{}'.format(int(layer_group) - 2,
      layer_block, layer_type) + tf_name
  return prefix + tf_name


def create_init_op(params):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  for var in variables:
    #name_split = var.name.split('/')
    #if len(name_split) != 3:
    #  continue
    #name = name_split[1] + '/' + name_split[2][:-2]
    name = var.name
    if name in params:
      #print(var.name, ' --> found init')
      init_map[var.name] = params[name]
      del params[name]
    else:
      print(var.name, ' --> init not found!')
      #raise 1
  print(list(params.keys()))
  #print(params['conv0/biases:0'].sum())
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


def build(inputs, labels, is_training, reuse=False):
  inputs = normalize_input(inputs)

  resnet_param = {}
  if reuse:
    tf.get_variable_scope().reuse_variables()
  MODEL_PATH ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(MODEL_DEPTH)+'.npy'
  param = np.load(MODEL_PATH, encoding='latin1').item()
  for k, v in param.items():
    try:
      newname = name_conversion(k)
    except:
      raise
    #print("Name Transform: " + k + ' --> ' + newname)
    resnet_param[newname] = v
    #print(v.shape)

  #logits = _build(image, is_training)
  #total_loss = loss(logits, labels, weights, num_labels, is_training)
  logits = _build(inputs, is_training)
  #total_loss = loss(logits, labels, is_training)

  init_op, init_feed = create_init_op(resnet_param)
  return logits, init_op, init_feed
  #if is_training:
  #  return [total_loss], init_op, init_feed
  #else:
  #  return [total_loss, logits, labels, img_names]

def minimize(opt, loss, global_step):
  #resnet_vars = tf.trainable_variables()
  all_vars = tf.trainable_variables()
  resnet_vars = []
  head_vars = []
  for v in all_vars:
    if v.name[:4] == 'head':
      print(v.name)
      head_vars += [v]
    else:
      resnet_vars += [v]
  grads_and_vars = opt.compute_gradients(loss, resnet_vars + head_vars)
  resnet_gv = grads_and_vars[:len(resnet_vars)]
  head_gv = grads_and_vars[len(resnet_vars):]
  lr_mul = 10
  #lr_mul = 1
  print(head_gv[0])
  #head_gv = [[g*lr_mul, v] for g,v in head_gv]
  print(head_gv[0])
  #  ygrad, _ = grads_and_vars[1]
  train_op = opt.apply_gradients(resnet_gv + head_gv, global_step=global_step)
  return train_op

  #my_vars = [the rest of variables]
  #opt1 = tf.train.GradientDescentOptimizer(0.00001)
  #opt2 = tf.train.GradientDescentOptimizer(0.0001)
  #grads = tf.gradients(loss, var_list1 + var_list2)
  #grads1 = grads[:len(var_list1)]
  #grads2 = grads[len(var_list1):]
  #tran_op1 = opt1.apply_gradients(zip(grads1, var_list1))
  #train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
  #train_op = tf.group(train_op1, train_op2)
  #x = tf.Variable(tf.ones([]))
  #y = tf.Variable(tf.zeros([]))
  #loss = tf.square(x-y)
  #global_step = tf.Variable(0, name="global_step", trainable=False)

  #  opt = tf.GradientDescentOptimizer(learning_rate=0.1)
  #  grads_and_vars = opt.compute_gradients(loss, [x, y])
  #  ygrad, _ = grads_and_vars[1]
  #  train_op = opt.apply_gradients([grads_and_vars[0], (ygrad*2, y)], global_step=global_step)

def loss(logits, logits2, labels, weights, num_labels, is_training=True):
  # TODO
  #loss_tf = tf.contrib.losses.softmax_cross_entropy()
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights)
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
  #w = 0.4
  #loss_val += w * losses.weighted_cross_entropy_loss(logits2, labels, weights, max_weight=10)
  
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=1)
  #loss_val = losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss_symmetric(logits, labels, weights, num_labels)
  all_losses = [loss_val]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss

def num_examples(dataset):
  return reader.num_examples(dataset)
