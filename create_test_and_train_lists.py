import numpy as np 
import glob, os
import pdb


def create_test_and_train_lists(num_train_images, num_test_images, dataset_STR, dataset_str, image_extension):
    if image_extension[0] == '.':
        image_extension = image_extension[1:]
    save_dir_to_write = "/root/singleshotpose/{}/{}/images".format(dataset_STR, dataset_str)
    path_to_output_lists = "./{}/{}".format(dataset_STR, dataset_str)
    path_to_images = "{}/images".format(path_to_output_lists)
    all_images = glob.glob(path_to_images + "/*.{}".format(image_extension))
    num_imgs = len(all_images)
    if num_imgs < num_test_images or num_imgs < num_train_images:
        raise RuntimeError("Not enough images!! ({} > {})".format(num_imgs, max(num_test_images, num_train_images)))
    elif num_imgs < num_train_images + num_test_images:
        print("WARNING: Not enough images for unique train / test sets... will reuse images for both sets!")

    img_inds = np.asarray(range(num_imgs))
    np.random.shuffle(img_inds)

    # Create and/or truncate train.txt and test.txt
    print("Saving test/train lists at: {}".format(path_to_output_lists))
    file_train = open(path_to_output_lists + '/train.txt', 'w')
    file_test  = open(path_to_output_lists + '/test.txt',  'w')

    if num_imgs < num_train_images + num_test_images: # will have to repeat
        for i in range(num_train_images):
            if i == num_train_images:
                break
            img_name = all_images[img_inds[i]].split('/')[-1]
            file_train.write(save_dir_to_write + "/" + img_name + "\n")
        np.random.shuffle(img_inds)
        for i in range(num_test_images):
            if i == num_test_images:
                break
            img_name = all_images[img_inds[i]].split('/')[-1]
            file_test.write(save_dir_to_write + "/" + img_name + "\n")
    else:
        # we have enough for both
        for i in range(num_train_images + num_test_images):
            img_name = all_images[img_inds[i]].split('/')[-1]
            if i < num_train_images:
                file_train.write(save_dir_to_write + "/" + img_name + "\n")
            else:
                file_test.write(save_dir_to_write + "/" + img_name + "\n")


    file_train.close()
    file_test.close()

if __name__ == "__main__":

    num_train_images = 1000  # 0 based indexing
    num_test_images = 1000  # 0 based indexing
    dataset_STR = 'MSL_QUAD'
    dataset_str = 'msl_quad'
    image_extension = 'png'
    
    create_test_and_train_lists(num_train_images, num_test_images, dataset_STR, dataset_str, image_extension)
    print("Finished!")
