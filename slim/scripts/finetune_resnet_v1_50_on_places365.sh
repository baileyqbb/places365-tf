#!/bin/bash
#
# This script performs the following operations:
# 
# 2. Fine-tunes a ResNetV1-50 model on the places365 training set.
# 3. Evaluates the model on the places365 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_places365.sh
set -e

# Only run the evaluation
ONLY_EVAL=True

# Where the pre-trained ResNetV1-50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/qianbb/Projects/places365-tf/models/pre-trained

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/qianbb/Projects/places365-tf/models/resnet_v1_50

# Where the dataset is saved to.
DATASET_DIR=/home/data/ILSVRC/place2/dataset_tfrecord/

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt ]; then
  wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
  tar -xvf resnet_v1_50_2016_08_28.tar.gz
  mv resnet_v1_50.ckpt ${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt
  rm resnet_v1_50_2016_08_28.tar.gz
fi

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=flowers \
#  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 3000 steps.
if ! ONLY_EVAL; then
  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=places365 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt \
    --checkpoint_exclude_scopes=resnet_v1_50/logits \
    --trainable_scopes=resnet_v1_50/logits \
    --max_number_of_steps=30000 \
    --batch_size=32 \
    --learning_rate=0.01 \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004 \
  #  --num_clones=4 \
  #  --clone_on_cpu=True
fi
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=places365 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50

if ! ONLY_EVAL; then
  # Fine-tune all the new layers for 1000 steps.
  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/all \
    --dataset_name=places365 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${TRAIN_DIR} \
    --model_name=resnet_v1_50 \
    --max_number_of_steps=10000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004 \
  #  --num_clones=4 \
  #  --clone_on_cpu=True
fi
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=places365 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50
