# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
	outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
	outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
	weight_decay: The l2 regularization coefficient.

  Returns:
	An arg_scope.
  """
  # Add normalizer_fn=slim.batch_norm if Batch Normalization is required!
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
					  activation_fn=tf.nn.relu,
					  weights_regularizer=slim.l2_regularizer(weight_decay),
					  biases_initializer=tf.zeros_initializer):
	with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
	  return arg_sc


def vgg_a(inputs,
		  num_classes=1000,
		  is_training=True,
		  dropout_keep_prob=0.5,
		  spatial_squeeze=True,
		  scope='vgg_a'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
	  net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	  net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
	  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	  net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
	  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout7')

	  net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')

	  # Convert end_points_collection into a end_point dict.
	  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	  if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	  return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	   net = slim.max_pool2d(net, [2, 2], scope='pool1')
	   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	   net = slim.max_pool2d(net, [2, 2], scope='pool2')
	   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
	   net = slim.max_pool2d(net, [2, 2], scope='pool3')
	   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
	   net = slim.max_pool2d(net, [2, 2], scope='pool4')
	   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
	   net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	   net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
	   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
	   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
						 scope='dropout7')
	   net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')
	   # Convert end_points_collection into a end_point dict.
	   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	   if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	   return net, end_points
vgg_16.default_image_size = 224

def vgg_16_face(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
		   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		   net = slim.max_pool2d(net, [2, 2], scope='pool1')
		   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		   net = slim.max_pool2d(net, [2, 2], scope='pool2')
		   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		   net = slim.max_pool2d(net, [2, 2], scope='pool3')
		   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		   net = slim.max_pool2d(net, [2, 2], scope='pool4')
		   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		   net = slim.max_pool2d(net, [2, 2], scope='pool5')
		  # Use conv2d instead of fully_connected layers.
		   net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
		   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
							 scope='dropout6')
		   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
		   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
							 scope='dropout7')
		   net = slim.conv2d(net, num_classes, [1, 1],
							activation_fn=None,
							normalizer_fn=None,
							scope='fc8')
		   # Convert end_points_collection into a end_point dict.
		   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
		   if spatial_squeeze:
			net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
			end_points[sc.name + '/fc8'] = net
		   return net, end_points
vgg_16_face.default_image_size = 224







def vgg_16_iris(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,normalizer_params={'is_training': is_training}):
	   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	   net = slim.max_pool2d(net, [2, 2], scope='pool1')
	   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	   net = slim.max_pool2d(net, [2, 2], scope='pool2')
	   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
	   net = slim.max_pool2d(net, [2, 2], scope='pool3')
	   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
	   net = slim.max_pool2d(net, [2, 2], scope='pool4')
	   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
	   net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	   net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
	   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
	   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
						 scope='dropout7')
	   net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8',weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
	   # Convert end_points_collection into a end_point dict.
	   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	   if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	   return net, end_points
vgg_16_iris.default_image_size = 224


def vgg_16_face_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
		   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		   net = slim.max_pool2d(net, [2, 2], scope='pool1')
		   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		   net = slim.max_pool2d(net, [2, 2], scope='pool2')
		   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		   net = slim.max_pool2d(net, [2, 2], scope='pool3')
		   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		   net = slim.max_pool2d(net, [2, 2], scope='pool4')
		   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		   net = slim.max_pool2d(net, [2, 2], scope='pool5')
		  # Use conv2d instead of fully_connected layers.
		   net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
		   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
							 scope='dropout6')
		   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
		   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
							 scope='dropout7')
		   net = slim.conv2d(net, num_classes, [1, 1],
							activation_fn=None,
							normalizer_fn=None,
							scope='fc8')
		   # Convert end_points_collection into a end_point dict.
		   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
		   if spatial_squeeze:
			net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
			end_points[sc.name + '/fc8'] = net
			net2=tf.squeeze(net1,[1,2],name='fc7/squeezed')
			end_points[sc.name + '/fc7'] = net2
			conv_value=tf.get_default_graph().get_tensor_by_name("vgg_19/fc7/convolution:0")
			conv_value=tf.squeeze(conv_value,[1,2],name='conv_value/squeezed')
		#	conv_value_1 = tf.get_default_graph().get_tensor_by_name("vgg_19/fc8/BiasAdd:0")
	#		conv_value_1 = tf.squeeze(conv_value, [1, 2], name='conv_value_1/squeezed')
			res = slim.get_model_variables()
		   return net, end_points,res,conv_value
vgg_16_face_feature_extract.default_image_size = 224







def vgg_16_iris_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,normalizer_params={'is_training': is_training}):
	   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	   net = slim.max_pool2d(net, [2, 2], scope='pool1')
	   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	   net = slim.max_pool2d(net, [2, 2], scope='pool2')
	   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
	   net = slim.max_pool2d(net, [2, 2], scope='pool3')
	   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
	   net = slim.max_pool2d(net, [2, 2], scope='pool4')
	   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
	   net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	   net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
	   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
	   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
						 scope='dropout7')
	   net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8',weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
	   # Convert end_points_collection into a end_point dict.
	   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	   if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
		net2=tf.squeeze(net1,[1,2],name='fc7/squeezed')
		end_points[sc.name + '/fc7'] = net2
		conv_value=tf.get_default_graph().get_tensor_by_name("vgg_16/fc7/convolution:0")
		conv_value=tf.squeeze(conv_value,[1,2],name='conv_value/squeezed')
		res = slim.get_model_variables()
	   return net, end_points,res,conv_value
vgg_16_iris_feature_extract.default_image_size = 224





def vgg_19(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	  net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
	  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	  net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
	  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout7')
	  net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')
	  # Convert end_points_collection into a end_point dict.
	  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	  if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	  return net, end_points
vgg_19.default_image_size = 224

def vgg_19_face_bilinear(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.

  Args:
	inputs: a tensor of size [batch_size, height, width, channels].
	num_classes: number of predicted classes.
	is_training: whether or not the model is being trained.
	dropout_keep_prob: the probability that activations are kept in the dropout
	  layers during training.
	spatial_squeeze: whether or not should squeeze the spatial dimensions of the
	  outputs. Useful to remove unnecessary dimensions for classification.
	scope: Optional scope for the variables.

  Returns:
	the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	  net = slim.conv2d(net, 128, [7, 7], padding='VALID', scope='fc6')
	  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	  net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')
	  # Convert end_points_collection into a end_point dict.
	  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	  if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	  return net, end_points
vgg_19_face_bilinear.default_image_size = 224

def vgg_19_face_gap(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	  net = slim.avg_pool2d(net, [7, 7], padding='VALID', scope='fc6')
	 # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
	 #					 scope='dropout6')
	  net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')
	  # Convert end_points_collection into a end_point dict.
	  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	  if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	  return net, end_points
vgg_19_face_gap.default_image_size = 224

def vgg_19_iris_gap(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
							normalizer_params={'is_training': is_training}):

			  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			  net = slim.max_pool2d(net, [2, 2], scope='pool1')
			  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			  net = slim.max_pool2d(net, [2, 2], scope='pool2')
			  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
			  net = slim.max_pool2d(net, [2, 2], scope='pool3')
			  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
			  net = slim.max_pool2d(net, [2, 2], scope='pool4')
			  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
			  net = slim.max_pool2d(net, [2, 2], scope='pool5')
			  # Use conv2d instead of fully_connected layers.
			  net = slim.avg_pool2d(net, [2, 16], padding='VALID', scope='fc6')
			  #  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
			#					 scope='dropout6')
			  net = slim.conv2d(net, num_classes, [1, 1],
								activation_fn=None,
								normalizer_fn=None,
								scope='fc8')
			  # Convert end_points_collection into a end_point dict.
			  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			  if spatial_squeeze:
				net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
				end_points[sc.name + '/fc8'] = net
			  return net, end_points
vgg_19_iris_gap.default_image_size = 224


def vgg_19_iris_bilinear(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
							normalizer_params={'is_training': is_training}):

			  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			  net = slim.max_pool2d(net, [2, 2], scope='pool1')
			  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			  net = slim.max_pool2d(net, [2, 2], scope='pool2')
			  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
			  net = slim.max_pool2d(net, [2, 2], scope='pool3')
			  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
			  net = slim.max_pool2d(net, [2, 2], scope='pool4')
			  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
			  net = slim.max_pool2d(net, [2, 2], scope='pool5')
			  # Use conv2d instead of fully_connected layers.
			  net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
			  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								 scope='dropout6')
			  net = slim.conv2d(net, num_classes, [1, 1],
								activation_fn=None,
								normalizer_fn=None,
								scope='fc8')
			  # Convert end_points_collection into a end_point dict.
			  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			  if spatial_squeeze:
				net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
				end_points[sc.name + '/fc8'] = net
			  return net, end_points
vgg_19_iris_bilinear.default_image_size = 224


def vgg_19_face_bilinear_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	  net1 = slim.conv2d(net, 64, [7, 7], padding='VALID', scope='fc6')
	  net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
						 scope='dropout6')
	  net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')
	  # Convert end_points_collection into a end_point dict.
	  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	  if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
		net2 = tf.squeeze(net1, [1, 2], name='fc6/squeezed')
		end_points[sc.name + '/fc6'] = net2
		conv_value = tf.get_default_graph().get_tensor_by_name("vgg_19/fc6/convolution:0")
		conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
		#	conv_value_1 = tf.get_default_graph().get_tensor_by_name("vgg_19/fc8/BiasAdd:0")
		#		conv_value_1 = tf.squeeze(conv_value, [1, 2], name='conv_value_1/squeezed')
		res = slim.get_model_variables()
	  return net, end_points, res, conv_value
vgg_19_face_bilinear_feature_extract.default_image_size = 224


def vgg_19_face_gap_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
	  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
	  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [2, 2], scope='pool5')
	  # Use conv2d instead of fully_connected layers.
	  net = slim.avg_pool2d(net, [7, 7], padding='VALID', scope='fc6')
	 # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
	 #					 scope='dropout6')
	  net = slim.conv2d(net, num_classes, [1, 1],
						activation_fn=None,
						normalizer_fn=None,
						scope='fc8')
	  # Convert end_points_collection into a end_point dict.
	  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
	  if spatial_squeeze:
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
		end_points[sc.name + '/fc8'] = net
	  return net, end_points
vgg_19_face_gap_feature_extract.default_image_size = 224


def vgg_19_iris_bilinear_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
							normalizer_params={'is_training': is_training}):

			  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			  net = slim.max_pool2d(net, [2, 2], scope='pool1')
			  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			  net = slim.max_pool2d(net, [2, 2], scope='pool2')
			  net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
			  net = slim.max_pool2d(net, [2, 2], scope='pool3')
			  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
			  net = slim.max_pool2d(net, [2, 2], scope='pool4')
			  net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
			  net = slim.max_pool2d(net, [2, 2], scope='pool5')
			  # Use conv2d instead of fully_connected layers.
			  net1 = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
			  net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
								 scope='dropout6')
			  net = slim.conv2d(net, num_classes, [1, 1],
								activation_fn=None,
								normalizer_fn=None,
								scope='fc8')
			  # Convert end_points_collection into a end_point dict.
			  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			  if spatial_squeeze:
				net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
				end_points[sc.name + '/fc8'] = net
			  net2 = tf.squeeze(net1, [1, 2], name='fc6/squeezed')
			  end_points[sc.name + '/fc6'] = net2
			  conv_value = tf.get_default_graph().get_tensor_by_name("vgg_19/fc6/convolution:0")
			  conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
			  #	conv_value_1 = tf.get_default_graph().get_tensor_by_name("vgg_19/fc8/BiasAdd:0")
			  #		conv_value_1 = tf.squeeze(conv_value, [1, 2], name='conv_value_1/squeezed')
			  res = slim.get_model_variables()
		return net, end_points, res, conv_value
vgg_19_iris_bilinear_feature_extract.default_image_size = 224




def vgg_19_face_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
		   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		   net = slim.max_pool2d(net, [2, 2], scope='pool1')
		   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		   net = slim.max_pool2d(net, [2, 2], scope='pool2')
		   net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
		   net = slim.max_pool2d(net, [2, 2], scope='pool3')
		   net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
		   net = slim.max_pool2d(net, [2, 2], scope='pool4')
		   net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
		   net = slim.max_pool2d(net, [2, 2], scope='pool5')
		  # Use conv2d instead of fully_connected layers.
		   net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
		   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
							 scope='dropout6')
		   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
		   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
							 scope='dropout7')
		   net = slim.conv2d(net, num_classes, [1, 1],
							activation_fn=None,
							normalizer_fn=None,
							scope='fc8')
		   # Convert end_points_collection into a end_point dict.
		   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
		   if spatial_squeeze:
			net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
			end_points[sc.name + '/fc8'] = net
			net2=tf.squeeze(net1,[1,2],name='fc7/squeezed')
			end_points[sc.name + '/fc7'] = net2
			conv_value=tf.get_default_graph().get_tensor_by_name("vgg_19/fc7/convolution:0")
			conv_value=tf.squeeze(conv_value,[1,2],name='conv_value/squeezed')
		#	conv_value_1 = tf.get_default_graph().get_tensor_by_name("vgg_19/fc8/BiasAdd:0")
	#		conv_value_1 = tf.squeeze(conv_value, [1, 2], name='conv_value_1/squeezed')
			res = slim.get_model_variables()
		   return net, end_points,res,conv_value
vgg_19_face_feature_extract.default_image_size = 224


def vgg_19_iris_feature_extract(inputs,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_19'):

  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
	end_points_collection = sc.name + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection):
		with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
		   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		   net = slim.max_pool2d(net, [2, 2], scope='pool1')
		   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		   net = slim.max_pool2d(net, [2, 2], scope='pool2')
		   net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
		   net = slim.max_pool2d(net, [2, 2], scope='pool3')
		   net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
		   net = slim.max_pool2d(net, [2, 2], scope='pool4')
		   net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
		   net = slim.max_pool2d(net, [2, 2], scope='pool5')
		  # Use conv2d instead of fully_connected layers.
		   net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
		   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
							 scope='dropout6')
		   net1 = slim.conv2d(net, 4096, [1, 1], scope='fc7')
		   net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
							 scope='dropout7')
		   net = slim.conv2d(net, num_classes, [1, 1],
							activation_fn=None,
							normalizer_fn=None,
							scope='fc8')
		   # Convert end_points_collection into a end_point dict.
		   end_points = slim.utils.convert_collection_to_dict(end_points_collection)
		   if spatial_squeeze:
			net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
			end_points[sc.name + '/fc8'] = net
			net2=tf.squeeze(net1,[1,2],name='fc7/squeezed')
			end_points[sc.name + '/fc7'] = net2
			conv_value=tf.get_default_graph().get_tensor_by_name("vgg_19/fc7/convolution:0")
			conv_value=tf.squeeze(conv_value,[1,2],name='conv_value/squeezed')
		#	conv_value_1 = tf.get_default_graph().get_tensor_by_name("vgg_19/fc8/BiasAdd:0")
	#		conv_value_1 = tf.squeeze(conv_value, [1, 2], name='conv_value_1/squeezed')
			res = slim.get_model_variables()
		   return net, end_points,res,conv_value
vgg_19_iris_feature_extract.default_image_size = 224

def vgg_19_joint(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				joint_feature = tf.concat(3, [net_face, net_iris])
				net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
				return net, end_points
vgg_19_joint.default_image_size = 224

def vgg_19_joint_feature_extract(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				joint_feature = tf.concat(3, [net_face, net_iris])
				net1 = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
					net2 = tf.squeeze(net1, [1, 2], name='fc_joint/squeezed')
					end_points[scope + '/fc_joint'] = net2
					conv_value = tf.get_default_graph().get_tensor_by_name("vgg_joint/fc_joint/convolution:0")
					conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
					res = slim.get_model_variables()
				return net, end_points, res, conv_value

vgg_19_joint_feature_extract.default_image_size = 224

def vgg_19_joint_1024(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				joint_feature = tf.concat(3, [net_face, net_iris])
				net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')
				net = slim.conv2d(net, 1024, [1, 1], scope='fc_joint_r')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint_r')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
				return net, end_points
vgg_19_joint_1024.default_image_size = 224

def vgg_19_joint_1024_feature_extract(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout6')
				net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				joint_feature = tf.concat(3, [net_face, net_iris])
				net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')
				net1 = slim.conv2d(net, 1024, [1, 1], scope='fc_joint_r')
				net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint_r')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
					net2 = tf.squeeze(net1, [1, 2], name='fc_joint_r/squeezed')
					end_points[scope + '/fc_joint_r'] = net2
					conv_value = tf.get_default_graph().get_tensor_by_name("vgg_joint/fc_joint_r/convolution:0")
					conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
					res = slim.get_model_variables()
				return net, end_points, res, conv_value

vgg_19_joint_1024_feature_extract.default_image_size = 224

def vgg_19_joint_bilinear(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


		#	with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
		#	                    normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
			#	net_face=slim.avg_pool2d(net,[7,7],scope='fc6')
				net_face = slim.conv2d(net, 64, [7, 7], activation_fn=None,padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				 #                  scope='dropout6')
				#net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net_iris = slim.conv2d(net, 64, [2, 16], activation_fn=None, padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout6')
				#net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				c = tf.matmul(net_face, net_iris, transpose_a=True)
				s_l = c.get_shape().as_list()
				joint_feature = tf.reshape(c, (-1, 1, 1, s_l[2]*s_l[3]))
				net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')



				#joint_feature = tf.concat(3, [net_face, net_iris])
				#net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout_joint')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
				return net, end_points
vgg_19_joint_bilinear.default_image_size = 224

def vgg_19_joint_128_bilinear(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


		#	with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
		#	                    normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
			#	net_face=slim.avg_pool2d(net,[7,7],scope='fc6')
				net_face = slim.conv2d(net, 128, [7, 7], activation_fn=None,padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				 #                  scope='dropout6')
				#net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net_iris = slim.conv2d(net, 128, [2, 16], activation_fn=None, padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout6')
				#net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				c = tf.matmul(net_face, net_iris, transpose_a=True)
				s_l = c.get_shape().as_list()
				joint_feature = tf.reshape(c, (-1, 1, 1, s_l[2]*s_l[3]))
				net = slim.conv2d(joint_feature, 8192, [1, 1], scope='fc_joint')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')
				net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint_r')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						   scope='dropout_joint_r')  #joint_feature = tf.concat(3, [net_face, net_iris])
				#net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout_joint')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
				return net, end_points
vgg_19_joint_128_bilinear.default_image_size = 224




def vgg_19_joint_bilinear_feature_extract(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


		#	with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
		#	                    normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net_face = slim.conv2d(net, 64, [7, 7], activation_fn=None,padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				 #                  scope='dropout6')
				#net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net_iris = slim.conv2d(net, 64, [2, 16], activation_fn=None, padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout6')
				#net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				c = tf.matmul(net_face, net_iris, transpose_a=True)
				s_l = c.get_shape().as_list()
				joint_feature = tf.reshape(c, (-1, 1, 1, s_l[2]*s_l[3]))
				#net1 = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				#net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
				#				   scope='dropout_joint')



				#joint_feature = tf.concat(3, [net_face, net_iris])

				net = slim.conv2d(joint_feature, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
					#net2 = tf.squeeze(net1, [1, 2], name='fc_joint/squeezed')
					#end_points[scope + '/fc_joint'] = net2
					#conv_value = tf.get_default_graph().get_tensor_by_name("vgg_joint/fc_joint/convolution:0")
					#conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
					#res = slim.get_model_variables()
					conv_value = tf.get_default_graph().get_tensor_by_name("vgg_joint/Reshape:0")
					conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
					res = slim.get_model_variables()
				return net, end_points, res, conv_value
vgg_19_joint_bilinear_feature_extract.default_image_size = 224


def vgg_19_joint_128_bilinear_feature_extract(inputs_face, inputs_iris,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.5,
				 spatial_squeeze=True,
				 scope='vgg_joint'):
	end_point = 'vgg_19_face'
	end_points_collection_face = scope + '_end_points'
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
						outputs_collections=end_points_collection_face):
		with tf.variable_scope(end_point, [inputs_face]) as sc:
			#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


		#	with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
		#	                    normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
			#	net_face=slim.avg_pool2d(net,[7,7],scope='fc6')
				net_face = slim.conv2d(net, 128, [7, 7], activation_fn=None,padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				 #                  scope='dropout6')
				#net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
			#				   scope='dropout7')
			# end_points[end_point]=net_face
		end_point = 'vgg_19_iris'
		with tf.variable_scope(end_point, [inputs_iris]) as sc:
			#	end_points_collection_iris = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#						outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				# Use conv2d instead of fully_connected layers.
				net_iris = slim.conv2d(net, 128, [2, 16], activation_fn=None, padding='VALID', scope='fc6')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout6')
				#net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
			# net_iris_drop = slim.dropout(net7face, dropout_keep_prob, is_training=is_training,
			#							 scope='dropout7')
			#	end_points[end_point] = net_iris
		end_point = 'vgg_joint'
		with tf.variable_scope(scope, 'vgg_joint', [net_face, net_iris]) as sc:
			# end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.

			# with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			#					outputs_collections=end_points_collection_face):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
								normalizer_params={'is_training': is_training}):
				c = tf.matmul(net_face, net_iris, transpose_a=True)
				s_l = c.get_shape().as_list()
				joint_feature = tf.reshape(c, (-1, 1, 1, s_l[2]*s_l[3]))
				net = slim.conv2d(joint_feature, 8192, [1, 1], scope='fc_joint')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								   scope='dropout_joint')
				net1 = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint_r')
				net = slim.dropout(net1, dropout_keep_prob, is_training=is_training,
						   scope='dropout_joint_r')  #joint_feature = tf.concat(3, [net_face, net_iris])
				#net = slim.conv2d(joint_feature, 4096, [1, 1], scope='fc_joint')
				#net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
				#                   scope='dropout_joint')
				net = slim.conv2d(net, num_classes, [1, 1],
								  activation_fn=None,
								  normalizer_fn=None,
								  scope='fc8_joint')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)
				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
					end_points[scope + '/fc8_joint'] = net
					net2 = tf.squeeze(net1, [1, 2], name='fc_joint_r/squeezed')
					end_points[scope + '/fc_joint_r'] = net2
					conv_value = tf.get_default_graph().get_tensor_by_name("vgg_joint/fc_joint_r/convolution:0")
					conv_value = tf.squeeze(conv_value, [1, 2], name='conv_value/squeezed')
					res = slim.get_model_variables()
				return net, end_points, res, conv_value
vgg_19_joint_128_bilinear_feature_extract.default_image_size = 224


def vgg_16_joint(inputs_face,inputs_iris,
		   num_classes=1000,
		   is_training=True,
		   dropout_keep_prob=0.5,
		   spatial_squeeze=True,
		   scope='vgg_joint'):

		end_point = 'vgg_19'
		end_points_collection_face = scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
							outputs_collections=end_points_collection_face):
			with tf.variable_scope(end_point, [inputs_face]) as sc:
		#		end_points_collection_face = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.


					with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
										normalizer_params={'is_training': is_training}):
						net = slim.repeat(inputs_face, 2, slim.conv2d, 64, [3, 3], scope='conv1')
						net = slim.max_pool2d(net, [2, 2], scope='pool1')
						net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
						net = slim.max_pool2d(net, [2, 2], scope='pool2')
						net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
						net = slim.max_pool2d(net, [2, 2], scope='pool3')
						net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
						net = slim.max_pool2d(net, [2, 2], scope='pool4')
						net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
						net = slim.max_pool2d(net, [2, 2], scope='pool5')
						# Use conv2d instead of fully_connected layers.
						net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
						net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
										   scope='dropout6')
						net_face = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
						#net_face_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
						#				   scope='dropout7')
		#end_points[end_point]=net_face
			end_point = 'vgg_16'
			with tf.variable_scope(end_point, [inputs_iris]) as sc:
				#	end_points_collection_iris = sc.name + '_end_points'
				# Collect outputs for conv2d, fully_connected and max_pool2d.
				#with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
				#						outputs_collections=end_points_collection_face):
					with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
									normalizer_params={'is_training': is_training}):
						net = slim.repeat(inputs_iris, 2, slim.conv2d, 64, [3, 3], scope='conv1')
						net = slim.max_pool2d(net, [2, 2], scope='pool1')
						net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
						net = slim.max_pool2d(net, [2, 2], scope='pool2')
						net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
						net = slim.max_pool2d(net, [2, 2], scope='pool3')
						net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
						net = slim.max_pool2d(net, [2, 2], scope='pool4')
						net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
						net = slim.max_pool2d(net, [2, 2], scope='pool5')
						# Use conv2d instead of fully_connected layers.
						net = slim.conv2d(net, 4096, [2, 16], padding='VALID', scope='fc6')
						net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
										   scope='dropout6')
						net_iris = slim.conv2d(net, 4096, [1, 1], activation_fn=None, scope='fc7')
						#net_iris_drop = slim.dropout(net_face, dropout_keep_prob, is_training=is_training,
						#							 scope='dropout7')
		#	end_points[end_point] = net_iris
			end_point = 'vgg_joint'
			with tf.variable_scope(scope, 'vgg_joint', [net_face,net_iris]) as sc:
				#end_points_collection = sc.name + '_end_points'
				# Collect outputs for conv2d, fully_connected and max_pool2d.

				#with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
				#					outputs_collections=end_points_collection_face):
					with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
									normalizer_params={'is_training': is_training}):

						joint_feature=tf.concat(3,[net_face, net_iris])
						net = slim.conv2d(joint_feature, 4096, [1, 1], normalizer_fn=None, scope='fc_joint')
						net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
									   scope='dropout_joint')
						net = slim.conv2d(net, num_classes, [1, 1],
									  activation_fn=None,
									  normalizer_fn=None,
									  scope='fc8_joint')

					end_points = slim.utils.convert_collection_to_dict(end_points_collection_face)

					if spatial_squeeze:
							net = tf.squeeze(net, [1, 2], name='fc8_joint/squeezed')
							end_points[scope+ '/fc8_joint'] = net
						#	end_points[sc.name + '/fc8_joint'] = net
					return net, end_points
vgg_16_joint.default_image_size = 224



# Alias
vgg_d = vgg_16
vgg_e = vgg_19
