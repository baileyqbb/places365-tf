#!/bin/bash
#
# This script performs the following operations:
# 1.
# 2. Trains a xception model on the places365 training set.
# 3. Evaluates the model on the places365 testing set.
#
# Usage:
# cd slim
# ./scripts/train_xception_on_places365.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/home/qianbb/PycharmProjects/places365-tf/models/xcpetion

# Where the dataset is saved to.
DATASET_DIR=/home/qianbb/data/indoorCVPR_09/Images/train_eval_tfrecord

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=cifar10 \
#  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=indoorCVPR_09 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=xception \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.045 \
  --learning_rate_decay_factor=0.94 \
  --num_epochs_per_decay=2 \
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=indoorCVPR_09 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=xception
