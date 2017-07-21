import sys
import numpy as np
import os
import tensorflow as tf

from time import clock
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import indoorCVPR_09
from datasets import places365

#######################
# Input Image(s) #
#######################
tf.app.flags.DEFINE_string(
    'image_file', '', 'Input image file name.')

#######################
# Model Flags #
#######################
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint for prediction.')

#######################
# Dataset Flags #
#######################
tf.app.flags.DEFINE_string('dataset_name', 'places365', 'Dataset name')

tf.app.flags.DEFINE_integer(
    'num_classes', 365, 'The number of predicted classes. This value should be matched with the pre-trained model loaded'
                        ' from the checkpoint_path.')

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

def predictor(image):
    #with tf.Graph().as_default():

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    ######################
    # Select the network #
    ######################
    #print(FLAGS.model_name)

    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=FLAGS.num_classes,
        is_training=False)


    image_size = network_fn.default_image_size

    processed_image = image_preprocessing_fn(image, image_size, image_size)

    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model
    logits, _ = network_fn(processed_images)

    probabilities = tf.nn.softmax(logits)

    #print(FLAGS.checkpoint_path)
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        slim.get_model_variables())

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init_fn(sess)

        start = clock()
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        finish = clock()
        print('Prediction time cost: %0.4f s' % ((finish - start)))
        #print((finish - start) / 1000)

        probabilities = probabilities[0, 0:]
        sorted_indx = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]

    #names = places365.create_readable_names_for_places365_labels()
    names = indoorCVPR_09.create_readable_names_for_indoorCVPR_09_labels()
    for i in range(5):
        indx = sorted_indx[i]
        print('Probability %0.2f => classs_name=%s' %
              (probabilities[indx], names[indx]))


def main(_):
    image = tf.image.decode_jpeg(tf.read_file(FLAGS.image_file),
                                 channels=3)
    predictor(image)


if __name__ == '__main__':
    tf.app.run()