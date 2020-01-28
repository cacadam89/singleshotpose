import numpy as np 
import glob, os
import pdb

num_train_images = 1000  # 0 based indexing
num_test_images = 1000  # 0 based indexing
path_to_output_lists = './MSLQUAD/mslquad/'
path_to_images = '/root/singleshotpose/MSLQUAD/mslquad/images/'
image_extension = 'png'

all_images = glob.glob(os.path.join(path_to_images, "*.{}".format(image_extension)))
num_imgs = len(all_images)
if num_imgs < num_test_images or num_imgs < num_train_images:
    raise RuntimeError("Not enough images!! ({} images)".format(num_imgs))
elif num_imgs < num_train_images + num_test_images:
    raise RuntimeWarning("Not enough images for unique train / test sets")

img_inds = np.asarray(range(num_imgs))
np.random.shuffle(img_inds)

# Create and/or truncate train.txt and test.txt
file_train = open(path_to_output_lists + 'train.txt', 'w')
file_test  = open(path_to_output_lists + 'test.txt',  'w')

if num_imgs < num_train_images + num_test_images: # will have to repeat
    for i in range(num_train_images):
        if i == num_train_images:
            break
        file_train.write(all_images[img_inds[i]] + "\n")
    np.random.shuffle(img_inds)
    for i in range(num_test_images):
        if i == num_test_images:
            break
        file_test.write(all_images[img_inds[i]] + "\n")
else:
    # we have enough for both
    for i in range(num_train_images + num_test_images):
        if i < num_train_images:
            file_train.write(all_images[img_inds[i]] + "\n")
        else:
            file_test.write(all_images[img_inds[i]] + "\n")

file_train.close()
file_test.close()