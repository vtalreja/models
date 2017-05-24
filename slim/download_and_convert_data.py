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
# ==============================================================================
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/casia
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import download_and_convert_casia
from datasets import download_and_convert_wvu_face_2013
from datasets import download_and_convert_wvu_iris_2013
from datasets import download_and_convert_wvu_face_overlap_2013
from datasets import download_and_convert_wvu_face_overlap_2012
from datasets import download_and_convert_wvu_iris_overlap_2013
from datasets import download_and_convert_wvu_iris_overlap_2012
from datasets import download_and_convert_wvu_face_overlap_frontal_2012
from datasets import download_and_convert_wvu_joint_face_overlap_2012_no_repeat
from datasets import download_and_convert_wvu_joint_iris_overlap_2012_no_repeat
from datasets import download_and_convert_wvu_joint_face_overlap_2013_no_repeat
from datasets import download_and_convert_wvu_joint_iris_overlap_2013_no_repeat
from datasets import download_and_convert_wvu_joint_iris_and_face_overlap_2013_no_repeat
from datasets import download_and_convert_wvu_joint_iris_and_face_overlap_2012_no_repeat
from datasets import download_and_convert_casia_ndiris

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    "wvu_iris_overlap_2012",
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    '/media/veerut/full/TensorFlow/Iris',
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'casia':
    download_and_convert_casia.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_face_2013':
      download_and_convert_wvu_face_2013.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_face_overlap_2013':
      download_and_convert_wvu_face_overlap_2013.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_face_overlap_2012':
      download_and_convert_wvu_face_overlap_2012.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_face_overlap_frontal_2012':
      download_and_convert_wvu_face_overlap_frontal_2012.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_joint_face_overlap_2012_no_repeat':
	  download_and_convert_wvu_joint_face_overlap_2012_no_repeat.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_joint_iris_overlap_2012_no_repeat':
      download_and_convert_wvu_joint_iris_overlap_2012_no_repeat.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_joint_face_overlap_2013_no_repeat':
      download_and_convert_wvu_joint_face_overlap_2013_no_repeat.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_joint_iris_overlap_2013_no_repeat':
      download_and_convert_wvu_joint_iris_overlap_2013_no_repeat.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_joint_iris_and_face_overlap_2013_no_repeat':
      download_and_convert_wvu_joint_iris_and_face_overlap_2013_no_repeat.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_joint_iris_and_face_overlap_2012_no_repeat':
      download_and_convert_wvu_joint_iris_and_face_overlap_2012_no_repeat.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_iris_2013':
      download_and_convert_wvu_iris_2013.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_iris_overlap_2013':
      download_and_convert_wvu_iris_overlap_2013.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'wvu_iris_overlap_2012':
      download_and_convert_wvu_iris_overlap_2012.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'casia_ndiris':
    download_and_convert_casia_ndiris.run(FLAGS.dataset_dir)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()

