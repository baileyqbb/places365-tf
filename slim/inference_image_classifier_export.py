# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from preprocessing import preprocessing_factory

FLAGS = None

class ImageClassifier:
    def __init__(self, label_file, model_name, graph_file, output_node):
        #self.image_file = image_file
        self.label_file = label_file
        self.model_name = model_name
        self.graph_file = graph_file
        self.output_node = output_node
        self.labels = self.load_labels()
        self.create_graph()
        self.sess = tf.Session()


    def load_labels(self):
      label = []
      proto_as_ascii_lines = tf.gfile.GFile(self.label_file).readlines()
      for l in proto_as_ascii_lines:
        label.append(l.rstrip())
      return label


    def create_graph(self):
      """Creates a graph from saved GraphDef file and returns a saver."""
      # C   reates graph from saved graph_def.pb.
      with tf.gfile.FastGFile(os.path.join(
              self.graph_file), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


    def read_tensor_from_image_file(self, image_file):
        input_name = "file_reader"
        file_reader = tf.read_file(image_file, input_name)
        if image_file.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels=3,
                                               name='png_reader')
        elif image_file.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                          name='gif_reader'))
        elif image_file.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                                name='jpeg_reader')

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            self.model_name, is_training=False)
        processed_image = image_preprocessing_fn(image_reader, 224, 224)
        processed_images = tf.expand_dims(processed_image, 0)
        sess = tf.Session()
        im_result = sess.run(processed_images)
        return im_result


    def run_inference_on_image(self, image_file):
        """Runs inference on an image.
        Args:
        image: Image file name.
        Returns:
        Nothing
        """
        if not tf.gfile.Exists(image_file):
          tf.logging.fatal('File does not exist %s', image_file)
          return -1

      #im_result = self.read_tensor_from_image_file(image_file)

      # Creates graph from saved GraphDef.
      #self.create_graph()

      #with tf.Session() as sess:
        im_result = self.read_tensor_from_image_file(image_file)
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        #names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        #print(names[-5:])
        input_tensor = self.sess.graph.get_operation_by_name('input')
        output_tensor = self.sess.graph.get_operation_by_name(self.output_node)
        predictions = self.sess.run(output_tensor.outputs[0],
                               {input_tensor.outputs[0]: im_result})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        #labels = self.load_labels()
        pred_results = []
        for node_id in top_k:
          human_string = self.labels[node_id]
          score = predictions[node_id]
          print('%s (score = %.5f)' % (human_string, score))
          pred_results.append([human_string, score])

        return pred_results
