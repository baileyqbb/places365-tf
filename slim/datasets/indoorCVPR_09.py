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
"""Provides data for the indoorCVPR_09 dataset.

The dataset scripts used to create the dataset can be found at:
build_train_image_data_indoorCVPR_09.py
NOTE: The original dataset contains some images with false image format which could not be read by the training program.
These images have been eliminated manually.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from six.moves import urllib

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*-of-00001'

SPLITS_TO_SIZES = {'train': 14231, 'validation': 1332}

_NUM_CLASSES = 67

LABEL_FILE = '/home/qianbb/data/indoorCVPR_09/indoorCVPR_09_labels.txt'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 1 and 365',
}

def create_readable_names_for_indoorCVPR_09_labels():
    """Create a dict mapping label id to human readable string.

      Returns:
          labels_to_names: dictionary where keys are integers from 1 to 365
          and values are human-readable names.
    """
    #filename = "../categories_places365.txt"
    #if not os.path.exists(filename):
    #    file_url = 'https://raw.githubusercontent.com/metalbubble/places_devkit/master/data/categories_places365.txt'
    #    filename = urllib.request.urlretrieve(file_url)

    if not os.path.exists(LABEL_FILE):
        print('Cannot find the label file: %s' % LABEL_FILE)
        return {}

    synset_list = [s.strip() for s in open(LABEL_FILE).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == _NUM_CLASSES

    synset_to_human = {}
    for s in synset_list:
        parts = s.strip().split(' ')
        assert len(parts) == 2
        synset = parts[0]
        indx = int(parts[1])
        synset_to_human[indx] = synset
    return synset_to_human


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
