# Coder: Qian Beibei
# Source paper: https://arxiv.org/pdf/1707.01083.pdf
# ==============================================================================
"""The main body of the shufflenet implemented in Tensorflow-slim.

Most part of the code is transplanted from the resnet_v2.py, which share the similar code architecture.

shufflenet is proposed in:
[1] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. arXiv:1707.01083


The key part of shufflenet compared with ResNet is the design of the grouped depth convolution and shuffle channel
for the bottleneck, which is similar with the ResNeXt, but with a manipulator of group number and the extra 'shuffle'
operation to the feature maps after the first grouped 1x1 convolution.

Typical use:

   from tensorflow.contrib.slim.nets import shufflenet

shufflenet for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(shufflenet.shufflenet_arg_scope()):
      net, end_points = shufflenet.shufflenet_50(inputs, 1000, groups=8, is_training=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_utils

slim = tf.contrib.slim
shufflenet_arg_scope = resnet_utils.resnet_arg_scope


def group_conv(inputs, noutputs, kernel=1, stride=1, scope=None):
    """
    Grouped convolution in the bottleneck. Separate the inputs into ngroups groups by channels and then do the convolution.
    :param inputs: 5D tensor in shape of [batch_size, input_height, input_width, ngroups, channels_per_group]
    :param noutputs: int. number of the outputs after the convolution
    :param kernel: int. the size of the kernal. Usually be 1 or 3
    :param stride: int. 1 or 2. If want to shrink th eimage size, then stride = 2
    :param scope: string. Scope
    :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel]
    """
    _, _, _, ngroups, _ = inputs.get_shape().as_list()

    shuffle_conv = []
    for i in range(ngroups):
        with tf.variable_scope(scope + '_group_%i' % i):
            input_group = inputs[:, :, :, i, :]
            #print('input_group_dim: {0}; noutputs:{1}; ngroups:{2}; kernel:{3}'.format(input_group.get_shape(), noutputs, ngroups, kernel))
            if kernel == 1:
                conv = slim.conv2d(input_group, noutputs, [1, 1], stride=1)
            elif kernel == 3:
                    conv = slim.separable_conv2d(input_group, noutputs, [3, 3], depth_multiplier=1, stride=stride, padding='SAME',
                                             normalizer_fn=slim.batch_norm, activation_fn=None, scope='DWConv')
                #conv = slim.separable_conv2d(input_group, noutputs, [3, 3], scope='DWConv')
            batch_size, height, width, channels = conv.get_shape().as_list()
            conv = tf.reshape(conv, [batch_size, height, width, 1, channels])
            shuffle_conv.append(conv)

    shuffle_conv = tf.concat(shuffle_conv, axis=3)

    return shuffle_conv


@slim.add_arg_scope
def bottleneck(inputs, depth, ngroups, stride, rate=1,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_shufflenet', [inputs]) as sc:
        batch_size, input_height, input_width, depth_in = inputs.get_shape().as_list()
        #print(batch_size, input_width, input_width, depth_in)
        if depth_in % ngroups != 0:
            ValueError('The group number needs to be divisible to the number of channels.')

        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        channel_per_group = depth_in // ngroups
        depth_per_group = depth // ngroups

        #separate the convolution
        net = tf.reshape(inputs, [batch_size, input_height, input_width, ngroups, channel_per_group])
        net = group_conv(net, channel_per_group, kernel=1, stride=1, scope='GConv1')
        #channel shuffle by transpose and flatten
        net = tf.transpose(net, [0, 1, 2, 4, 3])
        net = tf.reshape(net, [batch_size, input_height, input_width, -1])

        net = tf.reshape(net, [batch_size, input_height, input_width, ngroups, channel_per_group])
        net = group_conv(net, channel_per_group, kernel=3, stride=stride, scope='DWConv')

        net = group_conv(net, depth_per_group, kernel=1, stride=1, scope='GConv2')
        net = tf.reshape(net, [batch_size, input_height//stride, input_width//stride, -1])

        if stride == 1:
            net = shortcut + net
        elif stride == 2:
            shortcut = slim.avg_pool2d(shortcut, [3, 3], stride=2, padding='SAME')
            net = tf.concat([shortcut, net], axis=3)

        output = tf.nn.relu(net)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


def shufflenet(inputs,
               blocks,
               num_classes=None,
               is_training=True,
               global_pool=True,
               output_stride=None,
               include_root_block=True,
               spatial_squeeze=False,
               reuse=None,
               scope=None):
    """Generator for ResNeXt models.

    This function generates a family of shuffleNet models. See the shuffleNet_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths. Besides, most
    of the code is migrated from the resnet_v2.

    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNeXt
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNeXt block.

    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


    Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

    Raises:
    ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'shufflenet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                print('input name: %s' % net.name)
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # We do not include batch normalization or activation functions in
                    # conv1 because the first ResNet unit will perform these. Cf.
                    # Appendix of [2].
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        net = resnet_utils.conv2d_same(net, 24, 3, stride=2, scope='conv1')
                        #print('net name: %s' % net.name)
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                if spatial_squeeze:
                    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                else:
                    logits = net
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(logits, scope='predictions')

                #print('Scope name: %s' % sc.name)
                return logits, end_points
shufflenet.default_image_size = 224


def shufflenet_block(scope, base_depth, ngroups, num_units, stride):
  """Helper function for creating a resnext bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each group.
    cardinality: The number of the groups in the bottleneck
    bottleneck_type: The type of the bottleneck (b or c).
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnext bottleneck block.
  """

  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth,
      'ngroups': ngroups,
      'stride': stride
  }] + [{
      'depth': base_depth,
      'ngroups': ngroups,
      'stride': 1
  }] * (num_units - 1))
shufflenet.default_image_size = 224


def shufflenet_50_g4_d272(inputs,
                          num_classes=None,
                          is_training=True,
                          global_pool=True,
                          output_stride=None,
                          spatial_squeeze=True,
                          reuse=None,
                          scope='shufflenet_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      shufflenet_block('block1', base_depth=272, ngroups=4, num_units=4, stride=2),
      shufflenet_block('block2', base_depth=272*2, ngroups=4, num_units=8, stride=2),
      shufflenet_block('block3', base_depth=272*4, ngroups=4, num_units=4, stride=2),
  ]
  return shufflenet(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
shufflenet_50_g4_d272.default_image_size = shufflenet.default_image_size
