#!/bin/bash
#
# This script performs the following operations:
# 
# 2. Test the trained inception_resnet_v2 model on the places365 training set.
# 3. Evaluates the model on the places365 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/test_inception_resnet_v2_on_places365.sh
set -e

IMAGE_FILE=/home/qianbb/PycharmProjects/places365-tf/First_Student_IC_school_bus_202076.jpg

CHECKPOINT_PATH=/home/qianbb/PycharmProjects/places365-tf/models-indoorCVPR_09/mobilenet_v1_1_224/all/
#CHECKPOINT_PATH=/home/qianbb/Projects/places365-tf/models/pre-trained/inception_resnet_v2_2016_08_30.ckpt

python predict_image_classifier.py \
  --image_file=${IMAGE_FILE} \
  --num_classes=67 \
  --model_name=mobilenet_v1 \
  --preprocessing_name=mobilenet_v1 \
  --checkpoint_path=${CHECKPOINT_PATH}