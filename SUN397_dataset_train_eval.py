# Separate the SUN2012/SUN397 dataset into train and eval subsets according the given Partitions labels

import os
import glob
from shutil import copyfile, move

dataset_dir = "/home/data/SUN2012/"
image_folder = "SUN397/"
imageLable_dir = "Partitions/"

# Create the train and eval folder
train_dir = os.path.join(dataset_dir, image_folder, "train")
eval_dir = os.path.join(dataset_dir, image_folder, "eval")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Read the train labels and move the files to the train folder
def reform_dataset(setname):
    if setname == 'train':
        file_filter = 'Training*'
        _target_path = train_dir
    elif setname == 'eval':
        file_filter = 'Testing*'
        _target_path = eval_dir

    label_files = glob.glob(dataset_dir+imageLable_dir+file_filter)
    for label_file in label_files:
        file_list = [s.strip() for s in open(label_file).readlines()]

        for line in file_list:
             target_path = _target_path

             label_filename = line.split('/')

             label = label_filename[1]
             if not os.path.exists(os.path.join(target_path, label)):
                 os.makedirs(os.path.join(target_path, label))
             target_path = os.path.join(target_path, label)

             label = label_filename[2]
             if not os.path.exists(os.path.join(target_path, label)):
                 os.makedirs(os.path.join(target_path, label))
             target_path = os.path.join(target_path, label)


             if len(label_filename) > 4:
                 sub_label = label_filename[3]
                 if not os.path.exists(os.path.join(target_path, sub_label)):
                     os.makedirs(os.path.join(target_path, sub_label))
                 target_path = os.path.join(target_path, sub_label)

                 if len(label_filename) > 5:
                     sub_label2 = label_filename[4]
                     if not os.path.exists(os.path.join(target_path, sub_label2)):
                         os.makedirs(os.path.join(target_path, sub_label2))
                     target_path = os.path.join(target_path, sub_label2)

             orig_file_path = dataset_dir+image_folder+line
             dest_file_path = os.path.join(target_path, label_filename[-1])
             if os.path.exists(orig_file_path):
                 #  print("Moving %s to %s" % (orig_file_path, dest_file_path))
                 copyfile(orig_file_path, dest_file_path)

reform_dataset('train')
reform_dataset('eval')
