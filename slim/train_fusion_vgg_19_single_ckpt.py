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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import skimage.io as io
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
	'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
	'train_dir', '/media/veerut/DATADISK/TensorFlow/Joint/Joint_CheckPoints_2013_overlap_vgg_19/',
	'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
	'num_ps_tasks', 0,
	'The number of parameter servers. If the value is 0, then the parameters '
	'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
	'num_readers', 2,
	'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 10,
	'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
	'log_every_n_steps', 100,
	'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
	'save_summaries_secs', 1200,
	'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
	'save_interval_secs', 1200,
	'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
	'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
	'weight_decay', 0.0005, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
	'optimizer', 'sgd',
	'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
	'"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
	'adadelta_rho', 0.95,
	'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
	'adagrad_initial_accumulator_value', 0.1,
	'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
	'adam_beta1', 0.9,
	'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
	'adam_beta2', 0.999,
	'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
	'ftrl_initial_accumulator_value', 0.1,
	'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
	'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
	'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
	'momentum', 0.9,
	'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
	'learning_rate_decay_type',
	'exponential',
	'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
	' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
	'end_learning_rate', 0.0001,
	'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
	'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
	'learning_rate_decay_factor', 0.9, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
	'num_epochs_per_decay', 10.0,
	'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
	'sync_replicas', False,
	'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
	'replicas_to_aggregate', 1,
	'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
	'moving_average_decay', None,
	'The decay to use for the moving average.'
	'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
	'dataset_name_iris', 'wvu_joint_iris_overlap_2013_no_repeat', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
	'dataset_dir_iris',
	'/media/veerut/DATADISK/TensorFlow/Joint/Joint_Iris_Train_Data_2013_overlap_no_repeats_1_TfRecords',
	'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
	'model_name_iris', 'vgg_16_iris', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
	'preprocessing_name_iris', None, 'The name of the preprocessing to use. If left '
	                                 'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
	'New_Height_Of_Image_iris', 64, 'The Height of The Images in The Dataset. Default is 224')

tf.app.flags.DEFINE_integer(
	'New_Width_Of_Image_iris', 512, 'The Width of The Images in the dataset.Default is 224')

tf.app.flags.DEFINE_string(
	'dataset_name_face', 'wvu_joint_face_overlap_2013_no_repeat', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
	'dataset_dir_face',
	'/media/veerut/DATADISK/TensorFlow/Joint/Joint_Face_Train_Data_2013_overlap_no_repeats_1_TfRecords',
	'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
	'model_name_face', 'vgg_16_face', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
	'preprocessing_name_face', None, 'The name of the preprocessing to use. If left '
	                                 'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
	'New_Height_Of_Image_face', 224, 'The Height of The Images in The Dataset. Default is 224')

tf.app.flags.DEFINE_integer(
	'New_Width_Of_Image_face', 224, 'The Width of The Images in the dataset.Default is 224')

tf.app.flags.DEFINE_string(
	'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
	'model_name_joint', 'vgg_19_joint', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
	'labels_offset', 0,
	'An offset for the labels in the dataset. This flag is primarily used to '
	'evaluate the VGG and ResNet architectures which do not use a background '
	'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
	'batch_size', 30, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
	'train_image_size_iris', None, 'Train image size for iris')
tf.app.flags.DEFINE_integer(
	'train_image_size_face', None, 'Train image size for face')
tf.app.flags.DEFINE_integer('max_number_of_steps', 50000,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
	'checkpoint_path_iris', '/media/veerut/DATADISK/TensorFlow/Iris/CHKPNTIRIS13/model.ckpt-34744',
	'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
	'checkpoint_exclude_scopes_iris', 'vgg_19_iris/fc8/weights,vgg_19_iris/fc8/biases',
	'Comma-separated list of scopes of variables to exclude when restoring '
	'from a checkpoint.')

tf.app.flags.DEFINE_string(
	'trainable_scopes_iris', None,
	'Comma-separated list of scopes to filter the set of variables to train.'
	'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
	'ignore_missing_vars', True,
	'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_string(
	'checkpoint_path_face', '/media/veerut/DATADISK/TensorFlow/Face/CHKPNTFACE13/model.ckpt-52389',
	'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
	'checkpoint_exclude_scopes_face', 'vgg_19_face/fc8/weights,vgg_19_face/fc8/biases',
	'Comma-separated list of scopes of variables to exclude when restoring '
	'from a checkpoint.')

tf.app.flags.DEFINE_string(
	'trainable_scopes_face', None,
	'Comma-separated list of scopes to filter the set of variables to train.'
	'By default, None would train all the variables.')

tf.app.flags.DEFINE_string(
	'checkpoint_path_joint', None,
	'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
	'checkpoint_exclude_scopes_joint', None,
	'Comma-separated list of scopes of variables to exclude when restoring '
	'from a checkpoint.')

tf.app.flags.DEFINE_string(
	'trainable_scopes_joint', 'vgg_joint/fc_joint,vgg_joint/fc8_joint',
	'Comma-separated list of scopes to filter the set of variables to train.'
	'By default, None would train all the variables.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
	"""Configures the learning rate.

	Args:
	  num_samples_per_epoch: The number of samples in each epoch of training.
	  global_step: The global_step tensor.

	Returns:
	  A `Tensor` representing the learning rate.

	Raises:
	  ValueError: if
	"""
	decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
	                  FLAGS.num_epochs_per_decay)
	if FLAGS.sync_replicas:
		decay_steps /= FLAGS.replicas_to_aggregate

	if FLAGS.learning_rate_decay_type == 'exponential':
		return tf.train.exponential_decay(FLAGS.learning_rate,
		                                  global_step,
		                                  decay_steps,
		                                  FLAGS.learning_rate_decay_factor,
		                                  staircase=True,
		                                  name='exponential_decay_learning_rate')
	elif FLAGS.learning_rate_decay_type == 'fixed':
		return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
	elif FLAGS.learning_rate_decay_type == 'polynomial':
		return tf.train.polynomial_decay(FLAGS.learning_rate,
		                                 global_step,
		                                 decay_steps,
		                                 FLAGS.end_learning_rate,
		                                 power=1.0,
		                                 cycle=False,
		                                 name='polynomial_decay_learning_rate')
	else:
		raise ValueError('learning_rate_decay_type [%s] was not recognized',
		                 FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
	"""Configures the optimizer used for training.

	Args:
	  learning_rate: A scalar or `Tensor` learning rate.

	Returns:
	  An instance of an optimizer.

	Raises:
	  ValueError: if FLAGS.optimizer is not recognized.
	"""
	if FLAGS.optimizer == 'adadelta':
		optimizer = tf.train.AdadeltaOptimizer(
			learning_rate,
			rho=FLAGS.adadelta_rho,
			epsilon=FLAGS.opt_epsilon)
	elif FLAGS.optimizer == 'adagrad':
		optimizer = tf.train.AdagradOptimizer(
			learning_rate,
			initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
	elif FLAGS.optimizer == 'adam':
		optimizer = tf.train.AdamOptimizer(
			learning_rate,
			beta1=FLAGS.adam_beta1,
			beta2=FLAGS.adam_beta2,
			epsilon=FLAGS.opt_epsilon)
	elif FLAGS.optimizer == 'ftrl':
		optimizer = tf.train.FtrlOptimizer(
			learning_rate,
			learning_rate_power=FLAGS.ftrl_learning_rate_power,
			initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
			l1_regularization_strength=FLAGS.ftrl_l1,
			l2_regularization_strength=FLAGS.ftrl_l2)
	elif FLAGS.optimizer == 'momentum':
		optimizer = tf.train.MomentumOptimizer(
			learning_rate,
			momentum=FLAGS.momentum,
			name='Momentum')
	elif FLAGS.optimizer == 'rmsprop':
		optimizer = tf.train.RMSPropOptimizer(
			learning_rate,
			decay=FLAGS.rmsprop_decay,
			momentum=FLAGS.rmsprop_momentum,
			epsilon=FLAGS.opt_epsilon)
	elif FLAGS.optimizer == 'sgd':
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	else:
		raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
	return optimizer


def _add_variables_summaries(learning_rate):
	summaries = []
	for variable in slim.get_model_variables():
		summaries.append(tf.histogram_summary(variable.op.name, variable))
	summaries.append(tf.scalar_summary('training/Learning Rate', learning_rate))
	return summaries


def _get_init_op():
	if ((FLAGS.checkpoint_path_face is None) and (FLAGS.checkpoint_path_iris is None)):
		return None

	# Warn the user if a checkpoint exists in the train_dir. Then we'll be
	# ignoring the checkpoint anyway.
	if tf.train.latest_checkpoint(FLAGS.train_dir):
		tf.logging.info(
			'Ignoring --checkpoint_path because a checkpoint already exists in %s'
			% FLAGS.train_dir)
		return None, None

	exclusions_face = []
	if FLAGS.checkpoint_exclude_scopes_face:
		exclusions_face = [scope.strip()
		                   for scope in FLAGS.checkpoint_exclude_scopes_face.split(',')]

	exclusions_iris = []
	if FLAGS.checkpoint_exclude_scopes_face:
		exclusions_iris = [scope.strip()
		                   for scope in FLAGS.checkpoint_exclude_scopes_iris.split(',')]

	# TODO(sguada) variables.filter_variables()
	variables_to_restore_face = []
	for var in slim.get_model_variables('vgg_19_face'):
		excluded = False
		for exclusion in exclusions_face:
			if var.op.name.startswith(exclusion):
				excluded = True
				break
		if not excluded:
			variables_to_restore_face.append(var)

	variables_to_restore_iris = []
	for var in slim.get_model_variables('vgg_19_iris'):
		excluded = False
		for exclusion in exclusions_iris:
			if var.op.name.startswith(exclusion):
				excluded = True
				break
		if not excluded:
			if not var.op.name.startswith('vgg_16_joint'):
				variables_to_restore_iris.append(var)

	if tf.gfile.IsDirectory(FLAGS.checkpoint_path_face):
		checkpoint_path_face = tf.train.latest_checkpoint(FLAGS.checkpoint_path_face)
	else:
		checkpoint_path_face = FLAGS.checkpoint_path_face

	if tf.gfile.IsDirectory(FLAGS.checkpoint_path_iris):
		checkpoint_path_iris = tf.train.latest_checkpoint(FLAGS.checkpoint_path_iris)
	else:
		checkpoint_path_iris = FLAGS.checkpoint_path_iris

	tf.logging.info(
		'Fine-tuning from face checkpoint %s and iris checkpoint %s' % (checkpoint_path_face, checkpoint_path_iris))

	assign_op, feed_dict = assign_from_checkpoint_fusion(
		checkpoint_path_face,
		variables_to_restore_face, checkpoint_path_iris, variables_to_restore_iris)

	#	init_op = control_flow_ops.group(*[tf_variables.global_variables_initializer(), assign_op],
	#												 name='init_op')

	return assign_op, feed_dict


#	return init_op, feed_dict



def _get_variables_to_train():
	"""Returns a list of variables to train.

	Returns:
	  A list of variables to train by the optimizer.
	"""
	scopes = []
	variables_to_train_scope = []
	if FLAGS.trainable_scopes_iris is not None:
		scopes_iris = [scope.strip() for scope in FLAGS.trainable_scopes_iris.split(',')]
		for scope in scopes_iris:
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
			variables_to_train_scope.extend(variables)

	if FLAGS.trainable_scopes_face is not None:
		scopes_face = [scope.strip() for scope in FLAGS.trainable_scopes_face.split(',')]
		for scope in scopes_face:
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
			variables_to_train_scope.extend(variables)

	if FLAGS.trainable_scopes_joint is not None:
		scopes_joint = [scope.strip() for scope in FLAGS.trainable_scopes_joint.split(',')]
		for scope in scopes_joint:
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
			variables_to_train_scope.extend(variables)

	if ((FLAGS.trainable_scopes_iris is None) and (FLAGS.trainable_scopes_face is None) and (
		FLAGS.trainable_scopes_joint is None)):
		return tf.trainable_variables()

	return variables_to_train_scope


def assign_from_checkpoint_fusion(model_path_1, var_list_1, model_path_2, var_list_2):
	"""Creates an operation to assign specific variables from a checkpoint.

  Args:
	model_path: The full path to the model checkpoint. To get latest checkpoint
		use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
	var_list: A list of `Variable` objects or a dictionary mapping names in the
		checkpoint to the corresponding variables to initialize. If empty or
		None, it would return  no_op(), None.

  Returns:
	the restore_op and the feed_dict that need to be run to restore var_list.

  Raises:
	ValueError: If the checkpoint specified at `model_path` is missing one of
	  the variables in `var_list`.
  """
	reader_1 = pywrap_tensorflow.NewCheckpointReader(model_path_1)
	reader_2 = pywrap_tensorflow.NewCheckpointReader(model_path_2)

	if isinstance(var_list_1, (tuple, list)):
		var_list_1 = {var.op.name: var for var in var_list_1}

	if isinstance(var_list_2, (tuple, list)):
		var_list_2 = {var.op.name: var for var in var_list_2}

	feed_dict = {}
	assign_ops = []

	for checkpoint_var_name_1 in var_list_1:
		checkpoint_var_name_1_r = checkpoint_var_name_1.decode("utf-8").replace(u"vgg_19_face", "vgg_19")
		var = var_list_1[checkpoint_var_name_1]
		if not reader_1.has_tensor(checkpoint_var_name_1_r):
			raise ValueError(
				'Checkpoint is missing variable [%s]' % checkpoint_var_name_1_r)

		var_value = reader_1.get_tensor(checkpoint_var_name_1_r)
		placeholder_name = 'placeholder/' + var.op.name
		placeholder_value = array_ops.placeholder(
			dtype=var.dtype.base_dtype,
			shape=var.get_shape(),
			name=placeholder_name)
		assign_ops.append(var.assign(placeholder_value))

		if var.get_shape() != var_value.shape:
			raise ValueError(
				'Total size of new array must be unchanged for %s '
				'lh_shape: [%s], rh_shape: [%s]'
				% (checkpoint_var_name_1_r, str(var_value.shape), str(var.get_shape())))

		feed_dict[placeholder_value] = var_value.reshape(var.get_shape())

	for checkpoint_var_name_2 in var_list_2:
		checkpoint_var_name_2_r = checkpoint_var_name_2.decode("utf-8").replace(u"vgg_19_iris", "vgg_19")
		var = var_list_2[checkpoint_var_name_2]
		if not reader_2.has_tensor(checkpoint_var_name_2_r):
			raise ValueError(
				'Checkpoint is missing variable [%s]' % checkpoint_var_name_2_r)

		var_value = reader_2.get_tensor(checkpoint_var_name_2_r)
		placeholder_name = 'placeholder/' + var.op.name
		placeholder_value = array_ops.placeholder(
			dtype=var.dtype.base_dtype,
			shape=var.get_shape(),
			name=placeholder_name)
		assign_ops.append(var.assign(placeholder_value))

		if var.get_shape() != var_value.shape:
			raise ValueError(
				'Total size of new array must be unchanged for %s '
				'lh_shape: [%s], rh_shape: [%s]'
				% (checkpoint_var_name_2_r, str(var_value.shape), str(var.get_shape())))

		feed_dict[placeholder_value] = var_value.reshape(var.get_shape())

	assign_op = control_flow_ops.group(*assign_ops)
	return assign_op, feed_dict


def main(_):
	if ((not FLAGS.dataset_dir_iris) or (not FLAGS.dataset_dir_face)):
		raise ValueError('You must supply the dataset directory with --dataset_dir')

	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default():
		######################
		# Config model_deploy#
		######################
		deploy_config = model_deploy.DeploymentConfig(
			num_clones=FLAGS.num_clones,
			clone_on_cpu=FLAGS.clone_on_cpu,
			replica_id=FLAGS.task,
			num_replicas=FLAGS.worker_replicas,
			num_ps_tasks=FLAGS.num_ps_tasks)

		# Create global_step
		with tf.device(deploy_config.variables_device()):
			global_step = slim.create_global_step()

		######################
		# Select the dataset #
		######################
		dataset_iris = dataset_factory.get_dataset(
			FLAGS.dataset_name_iris, FLAGS.dataset_split_name, FLAGS.dataset_dir_iris)

		dataset_face = dataset_factory.get_dataset(
			FLAGS.dataset_name_face, FLAGS.dataset_split_name, FLAGS.dataset_dir_face)

		####################
		# Select the network #
		####################

		#  network_fn_iris = nets_factory.get_network_fn(
		#     FLAGS.model_name_iris,
		#    num_classes=(dataset.num_classes - FLAGS.labels_offset),
		#    weight_decay=FLAGS.weight_decay,
		#   is_training=True)

		network_fn_joint = nets_factory.get_network_fn_joint(
			FLAGS.model_name_joint,
			num_classes=(dataset_face.num_classes - FLAGS.labels_offset),
			weight_decay=FLAGS.weight_decay,
			is_training=True)

		#####################################
		# Select the preprocessing function #
		#####################################
		preprocessing_name_iris = FLAGS.preprocessing_name_iris or FLAGS.model_name_iris
		image_preprocessing_fn_iris = preprocessing_factory.get_preprocessing(
			preprocessing_name_iris,
			is_training=True)

		preprocessing_name_face = FLAGS.preprocessing_name_face or FLAGS.model_name_face
		image_preprocessing_fn_face = preprocessing_factory.get_preprocessing(
			preprocessing_name_face,
			is_training=True)

		##############################################################
		# Create a dataset provider that loads data from the dataset #
		##############################################################
		with tf.device(deploy_config.inputs_device()):
			provider_iris = slim.dataset_data_provider.DatasetDataProvider(
				dataset_iris,
				shuffle=False,
				num_readers=FLAGS.num_readers,
				common_queue_capacity=20 * FLAGS.batch_size,
				common_queue_min=10 * FLAGS.batch_size)
			[image_iris, label_iris] = provider_iris.get(['image', 'label'])
			label_iris -= FLAGS.labels_offset

			#	train_image_size_iris = FLAGS.train_image_size_iris or network_fn_iris.default_image_size
			new_height_iris = FLAGS.New_Height_Of_Image_iris or network_fn_iris.default_image_size
			new_width_iris = FLAGS.New_Width_Of_Image_iris or network_fn_iris.default_image_size

			#         image = image_preprocessing_fn(image, train_image_size, train_image_size)
			image_iris = image_preprocessing_fn_iris(image_iris, new_height_iris, new_width_iris)

			#  io.imshow(image)
			#  io.show()
			images_iris, labels_iris = tf.train.batch(
				[image_iris, label_iris],
				batch_size=FLAGS.batch_size,
				num_threads=FLAGS.num_preprocessing_threads,
				capacity=5 * FLAGS.batch_size)
			#      tf.image_summary('images', images)
			labels_iris = slim.one_hot_encoding(
				labels_iris, dataset_iris.num_classes - FLAGS.labels_offset)
			batch_queue_iris = slim.prefetch_queue.prefetch_queue(
				[images_iris, labels_iris], capacity=2 * deploy_config.num_clones)

		with tf.device(deploy_config.inputs_device()):
			provider_face = slim.dataset_data_provider.DatasetDataProvider(
				dataset_face,
				shuffle=False,
				num_readers=FLAGS.num_readers,
				common_queue_capacity=20 * FLAGS.batch_size,
				common_queue_min=10 * FLAGS.batch_size)
			[image_face, label_face] = provider_face.get(['image', 'label'])
			label_face -= FLAGS.labels_offset

			#	train_image_size_face = FLAGS.train_image_size_face or network_fn_face.default_image_size
			new_height_face = FLAGS.New_Height_Of_Image_face or network_fn_face.default_image_size
			new_width_face = FLAGS.New_Width_Of_Image_face or network_fn_face.default_image_size

			#         image = image_preprocessing_fn(image, train_image_size, train_image_size)
			image_face = image_preprocessing_fn_face(image_face, new_height_face, new_width_face)

			#  io.imshow(image)
			#  io.show()
			images_face, labels_face = tf.train.batch(
				[image_face, label_face],
				batch_size=FLAGS.batch_size,
				num_threads=FLAGS.num_preprocessing_threads,
				capacity=5 * FLAGS.batch_size)
			#      tf.image_summary('images', images)
			labels_face = slim.one_hot_encoding(
				labels_face, dataset_face.num_classes - FLAGS.labels_offset)
			batch_queue_face = slim.prefetch_queue.prefetch_queue(
				[images_face, labels_face], capacity=2 * deploy_config.num_clones)

		####################
		# Define the model #
		####################

		def clone_fn(batch_queue_iris, batch_queue_face):
			"""Allows data parallelism by creating multiple clones of network_fn."""
			images_iris, labels_iris = batch_queue_iris.dequeue()
			images_face, labels_face = batch_queue_face.dequeue()
			logits, end_points = network_fn_joint(images_face, images_iris)

			#  def clone_fn_face(batch_queue_face):
			#      """Allows data parallelism by creating multiple clones of network_fn."""
			#    images_face, labels_face = batch_queue_face.dequeue()
			#    logits_face, end_points_face, features_face,model_var_face = network_fn_face(images_face)







			#############################
			# Specify the loss function #
			#############################
			if 'AuxLogits' in end_points:
				slim.losses.softmax_cross_entropy(
					end_points['AuxLogits'], labels_face,
					label_smoothing=FLAGS.label_smoothing, weight=0.4, scope='aux_loss')
			slim.losses.softmax_cross_entropy(
				logits, labels_face, label_smoothing=FLAGS.label_smoothing, weight=1.0)

			# Adding the accuracy metric
			with tf.name_scope('accuracy'):
				predictions = tf.argmax(logits, 1)
				labels_face = tf.argmax(labels_face, 1)
				accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels_face)))
				tf.add_to_collection('accuracy', accuracy)
			return end_points

		# Gather initial summaries.
		summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

		clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue_iris, batch_queue_face])
		first_clone_scope = deploy_config.clone_scope(0)
		# Gather update_ops from the first clone. These contain, for example,
		# the updates for the batch_norm variables created by network_fn.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

		# Add summaries for end_points.
		end_points = clones[0].outputs
		for end_point in end_points:
			x = end_points[end_point]
			summaries.add(tf.histogram_summary('activations/' + end_point, x))
			summaries.add(tf.scalar_summary('sparsity/' + end_point,
			                                tf.nn.zero_fraction(x)))

		# Add summaries for losses.
		for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
			summaries.add(tf.scalar_summary('losses/%s' % loss.op.name, loss))

		# Add summaries for variables.
		for variable in slim.get_model_variables():
			summaries.add(tf.histogram_summary(variable.op.name, variable))

		#################################
		# Configure the moving averages #
		#################################
		if FLAGS.moving_average_decay:
			moving_average_variables = slim.get_model_variables()
			variable_averages = tf.train.ExponentialMovingAverage(
				FLAGS.moving_average_decay, global_step)
		else:
			moving_average_variables, variable_averages = None, None

		#########################################
		# Configure the optimization procedure. #
		#########################################
		with tf.device(deploy_config.optimizer_device()):
			learning_rate = _configure_learning_rate(dataset_face.num_samples, global_step)
			optimizer = _configure_optimizer(learning_rate)
			summaries.add(tf.scalar_summary('learning_rate', learning_rate,
			                                name='learning_rate'))

		if FLAGS.sync_replicas:
			# If sync_replicas is enabled, the averaging will be done in the chief
			# queue runner.
			optimizer = tf.train.SyncReplicasOptimizer(
				opt=optimizer,
				replicas_to_aggregate=FLAGS.replicas_to_aggregate,
				variable_averages=variable_averages,
				variables_to_average=moving_average_variables,
				replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
				total_num_replicas=FLAGS.worker_replicas)
		elif FLAGS.moving_average_decay:
			# Update ops executed locally by trainer.
			update_ops.append(variable_averages.apply(moving_average_variables))

		# Variables to train.
		variables_to_train = _get_variables_to_train()

		#  and returns a train_tensor and summary_op
		total_loss, clones_gradients = model_deploy.optimize_clones(
			clones,
			optimizer,
			var_list=variables_to_train)

		# # Add total_loss to summary.
		# summaries.add(tf.scalar_summary('total_loss', total_loss,
		#                                 name='total_loss'))

		# Add total_loss and accuacy to summary.
		summaries.add(tf.scalar_summary('eval/Total_Loss', total_loss,
		                                name='total_loss'))
		accuracy = tf.get_collection('accuracy', first_clone_scope)[0]
		summaries.add(tf.scalar_summary('eval/Accuracy', accuracy,
		                                name='accuracy'))

		# Create gradient updates.
		grad_updates = optimizer.apply_gradients(clones_gradients,
		                                         global_step=global_step)
		update_ops.append(grad_updates)

		update_op = tf.group(*update_ops)
		train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
		                                                  name='train_op')

		# Add the summaries from the first clone. These contain the summaries
		# created by model_fn and either optimize_clones() or _gather_clone_loss().
		summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
		                                   first_clone_scope))

		# Merge all summaries together.
		summary_op = tf.merge_summary(list(summaries), name='summary_op')

		init_iris, init_feed = _get_init_op()

		#	var_2=[v for v in tf.all_variables() if v.name == "vgg_19/conv3/conv3_3/weights:0"][0]


		###########################
		# Kicks off the training. #
		###########################
		slim.learning.train(
			train_tensor,
			logdir=FLAGS.train_dir,
			master=FLAGS.master,
			is_chief=(FLAGS.task == 0),
			init_fn=init_iris,
			init_feed_dict=init_feed,
			summary_op=summary_op,
			number_of_steps=FLAGS.max_number_of_steps,
			log_every_n_steps=FLAGS.log_every_n_steps,
			save_summaries_secs=FLAGS.save_summaries_secs,
			save_interval_secs=FLAGS.save_interval_secs,
			sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
	tf.app.run()
