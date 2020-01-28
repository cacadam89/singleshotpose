import numpy as np 
import pdb

num_train_images = 1000  # 0 based indexing
num_test_images = 1000  # 0 based indexing
path_to_save_lists = './MSLQUAD/mslquad/'
image_path_to_write = '/root/singleshotpose/MSLQUAD/mslquad/images/'
image_extension = 'png'

# Create and/or truncate train.txt and test.txt
file_train = open(path_to_save_lists + 'train.txt', 'w')
file_test  = open(path_to_save_lists + 'test.txt',  'w')

train_img_nums = np.asarray(range(num_train_images))
test_img_nums = np.asarray(range(num_test_images)) + num_train_images

# SHUFFLING IS DONE IN DATA LOADER:
# np.random.shuffle(train_img_nums)
# np.random.shuffle(test_img_nums)

for inum in train_img_nums:
    file_train.write("{}{:06d}.{}\n".format(image_path_to_write, inum, image_extension))

for inum in test_img_nums:
    file_test.write( "{}{:06d}.{}\n".format(image_path_to_write, inum, image_extension))

file_train.close()
file_test.close()