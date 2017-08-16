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
import glob

import numpy as np
from six.moves import urllib
import tensorflow as tf
from preprocessing import preprocessing_factory

FLAGS = None


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, FLAGS.graph_file), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def prepare_images(filepath):
    filetypes = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    filelist = []
    for ftype in filetypes:
        filelist.extend(glob.glob(os.path.join(filepath, ftype)))
    return filelist



def run_inference_on_image(image, labels, sess):
    """Runs inference on an image.
    Args:
    image: Image file name.
    Returns:
    Nothing
    """

    #image_data = tf.gfile.FastGFile(image, 'rb').read()

    image = tf.image.decode_jpeg(tf.read_file(image),
                           channels=3)

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    FLAGS.model_name, is_training=False)
    processed_image = image_preprocessing_fn(image, 224, 224)

    processed_images = tf.expand_dims(processed_image, 0)
    #sess = tf.Session()
    #im_result = sess.run(processed_images)

    # Creates graph from saved GraphDef.
    #create_graph()

    #with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    #names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    #print(names[-5:])

    im_result = sess.run(processed_images)

    input_tensor = sess.graph.get_operation_by_name(FLAGS.input_node_name)
    output_tensor = sess.graph.get_operation_by_name(FLAGS.output_node_name)
    predictions = sess.run(output_tensor.outputs[0],
                       {input_tensor.outputs[0]: im_result})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    #labels = load_labels(FLAGS.label_file)
    result = []
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
        result.append(human_string+'(%0.5f)' % score)

    return result



def main(_):
  # Prepare the output file
  output_file = open(FLAGS.output_to_file, 'w')
  output_file.write('Settings' + '\n')
  output_file.write('MODEL_DIR:'  + FLAGS.model_dir + '\n')
  output_file.write('MODEL_NAME:' + FLAGS.model_name + '\n')
  output_file.write('GRAPH_FILE:' + FLAGS.graph_file + '\n')
  output_file.write('OUTPUT_NODE:'+ FLAGS.output_node_name + '\n')
  output_file.write('LABEL_FILE:' + FLAGS.label_file + '\n')

  output_file.write('\nResults\n')

  create_graph()
  labels = load_labels(FLAGS.label_file)

  sess = tf.Session()

  if os.path.isdir(FLAGS.image_path):
      imagelist = prepare_images(FLAGS.image_path)
      indx = 1
      for image in imagelist:
          print('Inferencing: %d/%d' % (indx, len(imagelist)))
          output_file.write('IMAGE_NAME:' + image + '\n')
          results = run_inference_on_image(image, labels, sess)
          output_file.write('INFERENCE:%s\n' % results)
          indx += 1
  else:
      if not tf.gfile.Exists(FLAGS.image_path):
          tf.logging.fatal('File does not exist %s', FLAGS.image_path)
          return -1

      output_file.write('IMAGE_NAME:' + FLAGS.image_path + '\n')
      results = run_inference_on_image(FLAGS.image_path, labels, sess)
      output_file.write('INFERENCE:%s\n' % results)

  output_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--model_name',
      type=str,
      default='shufflenet_50_g4_d136',
      help="""\
      Model name.\
      """
  )
  parser.add_argument(
      '--graph_file',
      type=str,
      default='shufflenet_50_g4_d272_inf_graph_freeze.pb',
      help="""\
      Exported forzen graph file.\
      """
  )
  parser.add_argument(
      '--image_path',
      type=str,
      default='',
      help='Absolute path to image file.'
           'If the path is a folder, then load all the images under this folder'
  )
  parser.add_argument(
      '--label_file',
      type=str,
      default='',
      help='Absolute path to label file.'
  )
  parser.add_argument(
      '--input_node_name',
      type=str,
      default='input',
      help='Input node name.'
  )
  parser.add_argument(
      '--output_node_name',
      type=str,
      default='',
      help='Output node name.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  parser.add_argument(
      '--output_to_file',
      type=str,
      default='classification_results.txt',
      help='Save the classification results to file.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)