import numpy as np 
import pdb

num_images = 1000  # 0 based indexing
path_to_save_lists = './MSLQUAD_tiny/mslquad_tiny/'
image_path_to_write = '/root/singleshotpose/MSLQUAD_tiny/mslquad_tiny/images/'
image_extension = 'png'

# Create and/or truncate train.txt and test.txt
file_train = open(path_to_save_lists + 'train.txt', 'w')
file_test  = open(path_to_save_lists + 'test.txt',  'w')

img_nums = np.asarray(range(0, num_images))
np.random.shuffle(img_nums)

for inum in img_nums:
    file_train.write("{}{:06d}.{}\n".format(image_path_to_write, inum, image_extension))
    file_test.write( "{}{:06d}.{}\n".format(image_path_to_write, inum, image_extension))

file_train.close()
file_test.close()