# Coder: Qian Beibei
# Source paper: https://arxiv.org/abs/1611.05431
# ==============================================================================
"""The main body of the ResNeXt implemented in Tensorflow-slim.

Most part of the code is transplanted from the resnet_v2.py, which share the similar code architecture.

ResNeXt is proposed in:
[1] Saining Xie, Ross Girshick, Piotr Dollar, Zhouwen Tu, Kaiming He
    Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431


The key difference of ResNeXt compared with ResNet is the design of the multi-branch architecture for the bottleneck.
Specifically, instead of using a high-dimensional filter in the depth for the bottleneck, multiple low-dimensional
embeddings with the same topology for each layer in the block is aggregated. The number of the embeddings within the
bottleneck is called 'Cardinality'

Typical use:

   from tensorflow.contrib.slim.nets import resnext

ResNeXt-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnext.resnext_arg_scope()):
      net, end_points = resnext.resnext_101(inputs, 1000, is_training=False)

ResNeXt-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnext.resnet_arg_scope(is_training)):
      net, end_points = resnext.resnext_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_utils

slim = tf.contrib.slim
resnext_arg_scope = resnet_utils.resnet_arg_scope
"""
def conv_bn_relu_layer(inputs, depth, kernel, stride=1, relu=True):
    net = inputs
    net = slim.conv2d(net, depth, kernel, padding='SAME', stride=stride, scope='conv1')
    if relu:
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
    else:
        net = slim.batch_norm(net, activation_fn=None)
    return net
"""

def split(inputs, unit_depth, stride, rate=1):
    """
    The split structure in Figure 3b of the paper. It takes an input tensor. Conv it by [1, 1,
    64] filter, and then conv the result by [3, 3, 64]. Return the
    final resulted tensor, which is in shape of [batch_size, input_height, input_width, 64]
    :param inputs: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param unit_depth: the depth of each split
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel/64]
    """

    num_filter = unit_depth

    with tf.variable_scope('bneck_reduce_size'):
        conv = slim.conv2d(inputs, num_filter, [1, 1], stride=1)
    with tf.variable_scope('bneck_conv'):
        conv = resnet_utils.conv2d_same(conv, num_filter, 3, stride=stride, rate=rate)

    return conv


@slim.add_arg_scope
def bottleneck_b(inputs, unit_depth, cardinality, stride, rate=1,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_resnext_b', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        depth = unit_depth * cardinality * 2
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        split_list = []
        for i in range(cardinality):
            with tf.variable_scope('split_%i' % i):
                splits = split(inputs=preact, unit_depth=unit_depth, stride=stride, rate=rate)
            split_list.append(splits)

        # Concatenate splits and check the dimension
        concat_bottleneck = tf.concat(values=split_list, axis=3, name='bottleneck_concat')
        print('bottleneck_b--concat-dim:{}'.format(concat_bottleneck.get_shape()))
        net = slim.conv2d(concat_bottleneck, depth, [1, 1], stride=1, scope='bottleneck_conv3')
        print('shortcut-dim:{0}  net-dim:{1}'.format(shortcut.get_shape(), net.get_shape()))
        net = shortcut + net
        output = tf.nn.relu(net)
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


@slim.add_arg_scope
def bottleneck_c(inputs, unit_depth, cardinality, stride, rate=1,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_resnext_c', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        depth = unit_depth * cardinality * 2
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        net = slim.conv2d(inputs, unit_depth*cardinality, [1, 1], stride=1, scope='conv1')
        net = resnet_utils.conv2d_same(net, unit_depth*cardinality, 3, stride=stride, rate=rate, scope='grouped_conv2')
        net = slim.conv2d(net, depth, [1, 1], stride=1, scope='conv3')
        net = shortcut + net
        output = tf.nn.relu(net)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


def resnext(inputs,
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

    This function generates a family of ResNeXt models. See the resnext_*()
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
    with tf.variable_scope(scope, 'resnext', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck_b, bottleneck_c,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
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
                        net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
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
                return logits, end_points
resnext.default_image_size = 224


def resnext_block(scope, base_depth, cardinality, bottleneck_type, num_units, stride):
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
  if bottleneck_type == 'b':
      bottleneck = bottleneck_b
  elif bottleneck_type == 'c':
      bottleneck = bottleneck_c
  else:
      raise ValueError('Unknown type of the bottleneck. Should be type b or c.')

  return resnet_utils.Block(scope, bottleneck, [{
      'unit_depth': base_depth,
      'cardinality': cardinality,
      'stride': 1
  }] * (num_units - 1) + [{
      'unit_depth': base_depth,
      'cardinality': cardinality,
      'stride': stride
  }])
resnext.default_image_size = 224


def resnext_50(inputs,
               num_classes=None,
               width_bottleneck=4,
               cardinality=32,
               bottleneck_type='c',
               is_training=True,
               global_pool=True,
               output_stride=None,
               spatial_squeeze=True,
               reuse=None,
               scope='resnext_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnext_block('block1', base_depth=width_bottleneck, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=2),
      resnext_block('block2', base_depth=width_bottleneck*2, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=4, stride=2),
      resnext_block('block3', base_depth=width_bottleneck*4, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=6, stride=2),
      resnext_block('block4', base_depth=width_bottleneck*8, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=1),
  ]
  return resnext(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnext_50.default_image_size = resnext.default_image_size


def resnext_101(inputs,
                num_classes=None,
                width_bottleneck=4,
                cardinality=32,
                bottleneck_type='c',
                is_training=True,
                global_pool=True,
                output_stride=None,
                spatial_squeeze=True,
                reuse=None,
                scope='resnext_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnext_block('block1', base_depth=width_bottleneck, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=2),
      resnext_block('block2', base_depth=width_bottleneck*2, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=4, stride=2),
      resnext_block('block3', base_depth=width_bottleneck*4, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=23, stride=2),
      resnext_block('block4', base_depth=width_bottleneck*8, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=1),
  ]
  return resnext(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnext_101.default_image_size = resnext.default_image_size


def resnext_152(inputs,
                num_classes=None,
                width_bottleneck=4,
                cardinality=32,
                bottleneck_type='c',
                is_training=True,
                global_pool=True,
                output_stride=None,
                spatial_squeeze=True,
                reuse=None,
                scope='resnext_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnext_block('block1', base_depth=width_bottleneck, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=2),
      resnext_block('block2', base_depth=width_bottleneck*2, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=8, stride=2),
      resnext_block('block3', base_depth=width_bottleneck*4, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=36, stride=2),
      resnext_block('block4', base_depth=width_bottleneck*8, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=1),
  ]
  return resnext(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnext_152.default_image_size = resnext.default_image_size


def resnext_200(inputs,
                num_classes=None,
                width_bottleneck=4,
                cardinality=32,
                bottleneck_type='c',
                is_training=True,
                global_pool=True,
                output_stride=None,
                spatial_squeeze=True,
                reuse=None,
                scope='resnext_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      resnext_block('block1', base_depth=width_bottleneck, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=2),
      resnext_block('block2', base_depth=width_bottleneck*2, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=24, stride=2),
      resnext_block('block3', base_depth=width_bottleneck*4, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=36, stride=2),
      resnext_block('block4', base_depth=width_bottleneck*8, cardinality=cardinality, bottleneck_type=bottleneck_type,
                    num_units=3, stride=1),
  ]
  return resnext(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnext_200.default_image_size = resnext.default_image_size
