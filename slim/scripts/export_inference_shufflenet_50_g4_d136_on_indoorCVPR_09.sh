#!/bin/bash
#
# This script performs the following operations:
# 1. export the inference of the shufflenet_50_g4_d272 model trained on indoorCVPR_09 dataset
#
#
# Usage:
# cd slim
# ./scripts/export_inference_shufflenet_50_g4_d272_indoorCVPR_09.sh
set -e

# Where the exported inference are saved.
INFERENCE_GRAPH_PATH=/home/qianbb/PycharmProjects/places365-tf/models-indoorCVPR_09/shufflenet_50_g4_d136/all

# intermedia graph
TMP_GRAPH=shufflenet_50_g4_d136_inf_graph.pb

# final exported graph with trained weights
EXPORT_INF_GRAPH=shufflenet_50_g4_d136_inf_graph_freeze.pb

# Where the trained checkpoints are saved to
CHECKPOINT_PATH=/home/qianbb/PycharmProjects/places365-tf/models-indoorCVPR_09/shufflenet_50_g4_d136/all

# Where the dataset is saved to.
DATASET_DIR=/home/data/indoorCVPR_09/Images/train_eval_tfrecord/

# output node name
OUTPUT_NODE_NAME=shufflenet_50/predictions/Softmax

reExport=true

# export inference graph.
if [ -v reExport ]; then
    python export_inference_graph.py \
      --model_name=shufflenet_50_g4_d136 \
      --dataset_dir=${DATASET_DIR} \
      --dataset_name=indoorCVPR_09 \
      --output_file=${INFERENCE_GRAPH_PATH}/${TMP_GRAPH}
fi

# freeze graph to export the inference with the trained checkpoints.
if [ ${reExport+x} ]; then
    python ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
      --input_graph=${INFERENCE_GRAPH_PATH}/${TMP_GRAPH} \
      --input_checkpoint=${CHECKPOINT_PATH}/model.ckpt-273233 \
      --input_binary=true \
      --output_graph=${CHECKPOINT_PATH}/${EXPORT_INF_GRAPH} \
      --output_node_names=${OUTPUT_NODE_NAME}
fi

# test the exported graph
python ~/tensorflow/tensorflow/examples/label_image/label_image.py \
  --image=/home/qianbb/PycharmProjects/places365-tf/livingroom.jpg \
  --input_layer=input \
  --output_layer=${OUTPUT_NODE_NAME} \
  --graph=${CHECKPOINT_PATH}/${EXPORT_INF_GRAPH} \
  --labels=/home/qianbb/data/indoorCVPR_09/indoorCVPR_09_labels_noindx.txt \
  --input_mean=0 \
  --input_std=1 \
  --input_height=224 \
  --input_width=224
