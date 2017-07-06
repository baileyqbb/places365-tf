# Separate the indoorCVPR_09 dataset into train and eval subsets according the given TestImage labels

import os
from shutil import copyfile, move

dataset_dir = "/home/data/indoorCVPR_09/"
image_folder = "Images"
testImageLabel_file = "TestImages.txt"

# Create the train and eval folder
train_dir = os.path.join(dataset_dir, image_folder, "train")
eval_dir = os.path.join(dataset_dir, image_folder, "eval")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Read the test labels to replace the images
file_list = [s.strip() for s in open(testImageLabel_file).readlines()]

label_list = []
for line in file_list:
     label_filename = line.split('/')
     assert len(label_filename) == 2
     label = label_filename[0]
     label_list.append(label)
     filename = label_filename[1]
     if not os.path.exists(os.path.join(eval_dir, label)):
         os.makedirs(os.path.join(eval_dir, label))

     orig_file_path = os.path.join(dataset_dir, image_folder, line)
     dest_file_path = os.path.join(eval_dir, line)
     if os.path.exists(orig_file_path):
         #  print("Moving %s to %s" % (orig_file_path, dest_file_path))
         move(orig_file_path, dest_file_path)

# Move the rest images into the train folder
for folder in label_list:
    if not os.path.exists(os.path.join(train_dir, folder)):
        move(os.path.join(dataset_dir, image_folder, folder), train_dir)



