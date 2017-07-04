"""
# This script is to reorginize the validation dataset from the val_large to val_large_tf folder.
# In val_large_tf folder, the sub folders are named according to the class name of the validation images
"""
import sys
import os
import string
from shutil import copyfile
import numpy as np
import tensorflow as tf

datadir = "/home/data/ILSVRC/place2/"
val_dir = "val_large/"
val_target_dir = "val_large_tf"
val_labelfile = "places365_val.txt"
categories_labelfile = "categories_places365.txt"

# Create label folders
for folder in string.lowercase[:24]:
    if not os.path.exists(datadir+val_target_dir+folder):
        os.makedirs(datadir+val_target_dir+folder)

# Create subfolder according to the label file
unique_labels = [l.strip().split(' ')[0] for l in tf.gfile.FastGFile(
      datadir+categories_labelfile, 'r').readlines()]

for subfolder in unique_labels:
    if not os.path.exists(datadir+val_target_dir+subfolder):
        os.makedirs(datadir+val_target_dir+subfolder)

# Copy the images in the val_large dataset to the new folders
label_indx = range(len(unique_labels))
label_dict = {key:value for key, value in zip(label_indx, unique_labels)}

file_indx = [l.strip().split(' ') for l in tf.gfile.FastGFile(datadir+val_labelfile, 'r').readlines()]
for row in file_indx:
    filename = row[0]
    label_indx = row[1]
    target_folder = val_target_dir+label_dict[int(label_indx)]
    if not os.path.exists(datadir+target_folder+'/'+filename):
        copyfile(datadir+val_dir+filename, datadir+target_folder+'/'+filename)

