#!/bin/bash
#
# This script performs the following operations:
# 
# 2. Test the trained shufflenet_50_g4_d272 model on the indoorCVPR_09 training set.
# 3. Evaluates the model on the places365 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/test_shufflenet_50_g4_d272_on_indoorCVPR_09.sh
set -e

IMAGE_FILE=/home/qianbb/PycharmProjects/places365-tf/livingroom.jpg

CHECKPOINT_PATH=/home/qianbb/PycharmProjects/places365-tf/models-indoorCVPR_09/shufflenet_50_g4_d272/all/
#CHECKPOINT_PATH=/home/qianbb/Projects/places365-tf/models/pre-trained/inception_resnet_v2_2016_08_30.ckpt

python predict_image_classifier.py \
  --image_file=${IMAGE_FILE} \
  --num_classes=67 \
  --model_name=shufflenet_50_g4_d272 \
  --preprocessing_name=shufflenet_50_g4_d272 \
  --checkpoint_path=${CHECKPOINT_PATH}