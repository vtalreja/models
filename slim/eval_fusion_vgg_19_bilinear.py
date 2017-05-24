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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
	'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
	'max_num_batches', 700,
	'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
	'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
	'checkpoint_path', '/media/veerut/full/TensorFlow/Joint/Joint_CheckPoints_2013_overlap_vgg_19_joint_tf_record_no_bn_fc_layer_128_bilinear',
	'The directory where the model was written to or an absolute path to a '
	'checkpoint file.')

tf.app.flags.DEFINE_string(
	'eval_dir', '/media/veerut/full/TensorFlow/Joint/Joint_CheckPoints_2013_overlap_vgg_19_joint_tf_record_no_bn_fc_layer_128_bilinear_eval', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
	'eval_interval_secs', 60*2,
	'The frequency with which the model is evaluated, in seconds.')

tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 10,
	'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
	'dataset_name_joint', 'wvu_joint_iris_and_face_overlap_2012_no_repeat', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
	'dataset_split_name', 'train', 'The name of the train/testsplit.')

tf.app.flags.DEFINE_string(
	'dataset_dir_joint', '/media/veerut/full/TensorFlow/Joint/Joint_Iris_and_Face_Train_Data_2012_overlap_no_repeats_shuffle_TfRecords', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
	'labels_offset', 0,
	'An offset for the labels in the dataset. This flag is primarily used to '
	'evaluate the VGG and ResNet architectures which do not use a background '
	'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
	'model_name_face', 'vgg_19_face_bilinear', 'The name of the architecture to be used for preprocessing.')

tf.app.flags.DEFINE_string(
	'model_name_iris', 'vgg_19_iris_bilinear', 'The name of the architecture to be used for preprocessing.')

tf.app.flags.DEFINE_string(
	'model_name_joint', 'vgg_19_joint_128_bilinear', 'The name of the architecture to be used for preprocessing.')

tf.app.flags.DEFINE_string(
	'preprocessing_name', None, 'The name of the preprocessing to use. If left '
	'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
	'moving_average_decay', None,
	'The decay to use for the moving average.'
	'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
	'eval_image_size_face', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
	'New_Height_Of_Image_face', 224, 'The Height of The Images in The Dataset. Default is 224')

tf.app.flags.DEFINE_integer(
	'New_Width_Of_Image_face', 224, 'The Width of The Images in the dataset.Default is 224')

tf.app.flags.DEFINE_integer(
	'eval_image_size_iris', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
	'New_Height_Of_Image_iris', 64, 'The Height of The Images in The Dataset. Default is 224')

tf.app.flags.DEFINE_integer(
	'New_Width_Of_Image_iris', 512, 'The Width of The Images in the dataset.Default is 224')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if ((not FLAGS.dataset_dir_joint)):
	raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
	tf_global_step = slim.get_or_create_global_step()

	######################
	# Select the dataset #
	######################
	dataset_joint = dataset_factory.get_dataset(
		FLAGS.dataset_name_joint, FLAGS.dataset_split_name, FLAGS.dataset_dir_joint)


	####################
	# Select the model #
	####################
	network_fn = nets_factory.get_network_fn_joint(
		FLAGS.model_name_joint,
		num_classes=(dataset_joint.num_classes - FLAGS.labels_offset),
		is_training=False)

	##############################################################
	# Create a dataset provider that loads data from the dataset #
	##############################################################
	provider_joint = slim.dataset_data_provider.DatasetDataProvider(
		dataset_joint,
		shuffle=False,
		common_queue_capacity=20 * FLAGS.batch_size,
		common_queue_min=10 * FLAGS.batch_size)
	[image_iris, label_iris, image_face, label_face] = provider_joint.get(
		['image_iris', 'label_iris', 'image_face', 'label_face'])
	label_iris -= FLAGS.labels_offset
	label_face -= FLAGS.labels_offset

	#####################################
	# Select the preprocessing function #
	#####################################
	preprocessing_name_face = FLAGS.preprocessing_name or FLAGS.model_name_face
	image_preprocessing_fn_face = preprocessing_factory.get_preprocessing(
		preprocessing_name_face,
		is_training=False)

	eval_image_size_face = FLAGS.eval_image_size_face or network_fn.default_image_size
	new_height_face = FLAGS.New_Height_Of_Image_face or network_fn.default_image_size
	new_width_face = FLAGS.New_Width_Of_Image_face or network_fn.default_image_size


	image_face = image_preprocessing_fn_face(image_face, new_height_face, new_width_face)

	preprocessing_name_iris = FLAGS.preprocessing_name or FLAGS.model_name_iris
	image_preprocessing_fn_iris = preprocessing_factory.get_preprocessing(
		preprocessing_name_iris,
		is_training=False)

	eval_image_size_iris = FLAGS.eval_image_size_iris or network_fn.default_image_size
	new_height_iris = FLAGS.New_Height_Of_Image_iris or network_fn.default_image_size
	new_width_iris = FLAGS.New_Width_Of_Image_iris or network_fn.default_image_size


	image_iris = image_preprocessing_fn_iris(image_iris, new_height_iris, new_width_iris)

	images_iris, labels_iris, images_face, labels_face = tf.train.batch(
		[image_iris, label_iris, image_face, label_face],
		batch_size=FLAGS.batch_size,
		num_threads=FLAGS.num_preprocessing_threads,
		shapes=[(64, 512, 3), (), (224, 224, 3), ()],
		capacity=5 * FLAGS.batch_size)
	####################
	# Define the model #
	####################
	logits, endpoints = network_fn(images_face,images_iris)


	if FLAGS.moving_average_decay:
	  variable_averages = tf.train.ExponentialMovingAverage(
		  FLAGS.moving_average_decay, tf_global_step)
	  variables_to_restore = variable_averages.variables_to_restore(
		  slim.get_model_variables())
	  variables_to_restore[tf_global_step.op.name] = tf_global_step
	else:
	  variables_to_restore = slim.get_variables_to_restore()

	one_hot_labels = slim.one_hot_encoding(labels_face,  dataset_joint.num_classes - FLAGS.labels_offset)
	loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)

	predictions = tf.argmax(logits, 1)
	labels_face = tf.squeeze(labels_face)

	# Define the metrics:
	names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
		'Total_Loss': slim.metrics.streaming_mean(loss),
		'Accuracy': slim.metrics.streaming_accuracy(predictions, labels_face),
		'Recall@5': slim.metrics.streaming_recall_at_k(
			logits, labels_face, 5),
	})

	# Print the summaries to screen.
	for name, value in list(names_to_values.items()):
	  summary_name = 'eval/%s' % name
	  op = tf.scalar_summary(summary_name, value, collections=[])
	  op = tf.Print(op, [value], summary_name)
	  tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

	# TODO(sguada) use num_epochs=1
	if FLAGS.max_num_batches:
	  num_batches = FLAGS.max_num_batches
	else:
	  # This ensures that we make a single pass over all of the data.
	  num_batches = math.ceil(dataset_joint.num_samples / float(FLAGS.batch_size))

	# if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
	#   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
	# else:
	#   checkpoint_path = FLAGS.checkpoint_path

	  tf.logging.info('Evaluating %s' % FLAGS.checkpoint_path)


	slim.evaluation.evaluation_loop(
		master=FLAGS.master,
		checkpoint_dir=FLAGS.checkpoint_path,
		logdir=FLAGS.eval_dir,
		num_evals=num_batches,
		eval_op=list(names_to_updates.values()),
		variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
