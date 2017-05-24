
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # NOTE: THE ABOVE COPY WRITE LEFT UNTOUCH BECAUSE THIS CODE IS A MODIFIED VERSION OF THE MAIN FILE PROVIDED BY THE MAIN CONTRIBUTORS.
# ==============================================================================
r"""Converts CASIA data to TFRecords of TF-Example protos.

# This module does not download CASIA because it needs a license agreement.

This module reads the files of CASIA dataset and convert them into TFrecords.
It split the data into train & test.

Required raw data format:

					* All folders are separated pey subject.
					* All images in a folder are color images and relevant to one subject.

The script should take about 20 minutes to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION =0

# Seed for repeatability.
_RANDOM_SEED = 1

# The number of shards per dataset split(It is recommended to set as athe maximum number of supported cores).
_NUM_SHARDS = 1


class ImageReader(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self):
		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

	def read_image_dims(self, sess, image_data):
		image = self.decode_jpeg(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg,
						 feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image


def _get_filenames_and_classes(dataset_dir,folder_dir):
	"""Returns a list of filenames and inferred class names.

	Args:
	  dataset_dir: A directory containing a set of subdirectories representing
		class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
	  A list of image file paths, relative to `dataset_dir` and the list of
	  subdirectories, representing class names.
	"""
	flower_root = os.path.join(dataset_dir, folder_dir)
	directories = []
	class_names = []
	for filename in os.listdir(flower_root):
		path = os.path.join(flower_root, filename)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(filename)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			path = os.path.join(directory, filename)
			photo_filenames.append(path)

	return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'wvu_joint_iris_and_face_overlap_2012_no_repeat_%s_%05d-of-%05d.tfrecord' % (
		split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
	"""Converts the given filenames to a TFRecord dataset.

	Args:
	  split_name: The name of the dataset, either 'train' or 'validation'.
	  filenames: A list of absolute paths to png or jpg images.
	  class_names_to_ids: A dictionary from class names (strings) to ids
		(integers).
	  dataset_dir: The directory where the converted datasets are stored.
	"""
	assert split_name in ['train', 'validation']

	num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:

			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(
					dataset_dir, split_name, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i + 1, len(filenames), shard_id))
						sys.stdout.flush()

						# Read the filename:
						image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
						height, width = image_reader.read_image_dims(sess, image_data)

						class_name = os.path.basename(os.path.dirname(filenames[i]))
						class_id = class_names_to_ids[class_name]

						example = dataset_utils.image_to_tfexample(
							image_data, 'jpg', height, width, class_id)
						tfrecord_writer.write(example.SerializeToString())

	sys.stdout.write('\n')
	sys.stdout.flush()

def _convert_data(split_name, filenames_iris,filenames_face, class_names_to_ids_iris,class_names_to_ids_face, dataset_dir):
	"""Converts the given filenames to a TFRecord dataset.

	Args:
	  split_name: The name of the dataset, either 'train' or 'validation'.
	  filenames: A list of absolute paths to png or jpg images.
	  class_names_to_ids: A dictionary from class names (strings) to ids
		(integers).
	  dataset_dir: The directory where the converted datasets are stored.
	"""
	assert split_name in ['train', 'validation']

	num_per_shard = int(math.ceil(len(filenames_iris) / float(_NUM_SHARDS)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:

			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(
					dataset_dir, split_name, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id + 1) * num_per_shard, len(filenames_iris))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i + 1, len(filenames_iris), shard_id))
						sys.stdout.flush()

						# Read the filename:
						image_data_iris = tf.gfile.FastGFile(filenames_iris[i], 'r').read()
						image_data_face = tf.gfile.FastGFile(filenames_face[i], 'r').read()
						height_iris, width_iris = image_reader.read_image_dims(sess, image_data_iris)
						height_face, width_face = image_reader.read_image_dims(sess, image_data_face)

						class_name_iris = os.path.basename(os.path.dirname(filenames_iris[i]))
						class_name_face = os.path.basename(os.path.dirname(filenames_face[i]))


						class_id_iris = class_names_to_ids_iris[class_name_iris]
						class_id_face = class_names_to_ids_face[class_name_face]

						if(class_id_iris == class_id_face):
							example = image_example(
							image_data_iris,'jpg', height_iris, width_iris, class_id_iris,image_data_face,'jpg', height_face, width_face, class_id_face)
							tfrecord_writer.write(example.SerializeToString())
						else:
							raise ValueError('class id for iris %s is not equal to class id for face %s' % (
							class_id_iris, class_id_face))

	sys.stdout.write('\n')
	sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
	"""Removes temporary files used to create the dataset.

	Args:
	  dataset_dir: The directory where the temporary files are stored.
	"""
	filename = _DATA_URL.split('/')[-1]
	filepath = os.path.join(dataset_dir, filename)
	tf.gfile.Remove(filepath)

	tmp_dir = os.path.join(dataset_dir, 'CASIA-maxpy-clean')
	tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
	for split_name in ['train', 'validation']:
		for shard_id in range(_NUM_SHARDS):
			output_filename = _get_dataset_filename(
				dataset_dir, split_name, shard_id)
			if not tf.gfile.Exists(output_filename):
				return False
	return True

def image_example(image_data_iris,image_format_iris, height_iris, width_iris, class_id_iris,image_data_face,image_format_face, height_face, width_face, class_id_face):
  return tf.train.Example(features=tf.train.Features(feature={
	  'image_iris/encoded': dataset_utils.bytes_feature(image_data_iris),
	  'image_iris/format': dataset_utils.bytes_feature(image_format_iris),
	  'image_iris/class/label': dataset_utils.int64_feature(class_id_iris),
	  'image_iris/height': dataset_utils.int64_feature(height_iris),
	  'image_iris/width': dataset_utils.int64_feature(width_iris),
	  'image_face/encoded': dataset_utils.bytes_feature(image_data_face),
	  'image_face/format': dataset_utils.bytes_feature(image_format_face),
	  'image_face/class/label': dataset_utils.int64_feature(class_id_face),
	  'image_face/height': dataset_utils.int64_feature(height_face),
	  'image_face/width': dataset_utils.int64_feature(width_face),
  }))

def run(dataset_dir):
	"""Runs the download and conversion operation.

	Args:
	  dataset_dir: The dataset directory where the dataset is stored.
	"""
	if not tf.gfile.Exists(dataset_dir):
		tf.gfile.MakeDirs(dataset_dir)

	if _dataset_exists(dataset_dir):
		print('Dataset files already exist. Exiting without re-creating them.')
		return

	photo_filenames_iris, class_names_iris = _get_filenames_and_classes(dataset_dir,'/media/veerut/DATADISK/TensorFlow/Joint/Joint_Iris_Train_Data_2012_overlap_no_repeats')
	photo_filenames_face, class_names_face = _get_filenames_and_classes(dataset_dir,'/media/veerut/DATADISK/TensorFlow/Joint/Joint_Face_Train_Data_2012_overlap_no_repeats')

	class_names_to_ids_iris = dict(zip(class_names_iris, range(len(class_names_iris))))
	class_names_to_ids_face = dict(zip(class_names_face, range(len(class_names_face))))

	# Divide into train and test:
	random.seed(_RANDOM_SEED)
	random.shuffle(photo_filenames_iris)
	random.seed(_RANDOM_SEED)
	random.shuffle(photo_filenames_face)
	training_filenames_iris = photo_filenames_iris[_NUM_VALIDATION:]
	validation_filenames_iris = photo_filenames_iris[:_NUM_VALIDATION]
	training_filenames_face = photo_filenames_face[_NUM_VALIDATION:]
	validation_filenames_face = photo_filenames_face[:_NUM_VALIDATION]

	# First, convert the training and validation sets.
	_convert_data('train', training_filenames_iris, training_filenames_face, class_names_to_ids_iris,class_names_to_ids_face,
					 dataset_dir)
	_convert_data('validation', validation_filenames_iris, validation_filenames_face, class_names_to_ids_iris,class_names_to_ids_face,
					 dataset_dir)

	# Finally, write the labels file:
	print("len(class_names)",len(class_names_iris))
	labels_to_class_names = dict(zip(range(len(class_names_iris)), class_names_iris))
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

	# Uncomment if cleaning the data files is desired.
	# # _clean_up_temporary_files(dataset_dir)
	# _clean_up_temporary_files(dataset_dir)
	print('\nFinished converting the WVU Joint Iris Overlap 2012 dataset!')
