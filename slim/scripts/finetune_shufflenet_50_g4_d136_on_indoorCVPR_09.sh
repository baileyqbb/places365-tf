#!/bin/bash
#
# This script performs the following operations:
# 
# 2. Fine-tunes a shufflenet_50 model on the indoorCVPR_09 training set.
# 3. Evaluates the model on the indoorCVPR_09 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_shufflenet_50_g4_d272_on_indoorCVPR_09.sh
set -e

# Where the pre-trained shufflenet_50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/qianbb/Projects/places365-tf/models/shufflenet_50_g4_d136

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/qianbb/Projects/places365-tf/models-indoorCVPR_09/shufflenet_50_g4_d136

# Where the dataset is saved to.
DATASET_DIR=/home/data/indoorCVPR_09/Images/train_eval_tfrecord/

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
#if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt ]; then
#  wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
#  tar -xvf mobilenet_v1_1.0_224_2017_06_14.tar.gz
#  mv resnet_v1_50.ckpt ${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt
#  rm mobilenet_v1_1.0_224_2017_06_14.tar.gz
#fi

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=flowers \
#  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 3000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=indoorCVPR_09 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=shufflenet_50_g4_d136 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=shufflenet_50/logits \
  --trainable_scopes=shufflenet_50/logits\
  --max_number_of_steps=300000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --weight_decay=0.0004
#  --num_clones=1
#  --clone_on_cpu=True

# Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
# --eval_dir=${TRAIN_DIR} \
#  --dataset_name=indoorCVPR_09 \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_resnet_v2


# Fine-tune all the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=indoorCVPR_09 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=shufflenet_50_g4_d136 \
  --max_number_of_steps=300000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
#  --num_clones=1
#  --clone_on_cpu=True

# Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR}/all \
#  --eval_dir=${TRAIN_DIR}/all \
#  --dataset_name=indoorCVPR_09 \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_resnet_v2
