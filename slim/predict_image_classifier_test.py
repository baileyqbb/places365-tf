#from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2
import time

from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing

checkpoints_dir = '/home/qianbb/PycharmProjects/places365-tf/models/inception_resnet_v2/all'

image_dir = '/home/qianbb/PycharmProjects/places365-tf/qbb.jpg'#First_Student_IC_school_bus_202076.jpg'

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = 299 or inception_resnet_v2.default_image_size


with tf.Graph().as_default():

    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")

    # Open specified url and load image as a string
    #image_string = urllib2.urlopen(url).read()

    # Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(tf.read_file(image_dir), channels=3)

    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = inception_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images  = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images,
                               num_classes=365,
                               is_training=False)

    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)

    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    if tf.gfile.IsDirectory(checkpoints_dir):
        checkpoint_path = tf.train.latest_checkpoint(checkpoints_dir)
    else:
        checkpoint_path = checkpoints_dir

    #print(checkpoints_dir)
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        slim.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:

        # Load weights
        init_fn(sess)

        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
    """
    # Show the downloaded image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    # Show the image that is actually being fed to the network
    # The image was resized while preserving aspect ratio and then
    # cropped. After that, the mean pixel value was subtracted from
    # each pixel of that crop. We normalize the image to be between [-1, 1]
    # to show the image.
    plt.imshow( network_input / (network_input.max() - network_input.min()) )
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    """
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # Now we print the top-5 predictions that the network gives us with
        # corresponding probabilities. Pay attention that the index with
        # class names is shifted by 1 -- this is because some networks
        # were trained on 1000 classes and others on 1001. VGG-16 was trained
        # on 1000 classes.
        print('Probability %0.2f => [%s]' % (probabilities[index], index))

    res = slim.get_model_variables()
    print(res)