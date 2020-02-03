#!/usr/bin/python
# encoding: utf-8

import os
import random
from PIL import Image
import numpy as np
import numpy.linalg as la
from image import *
import torch

from torch.utils.data import Dataset
from utils import read_truths_args, read_truths, get_all_files, pnp
from raptor_camera import camera
import pdb


class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4, cell_size=32, bg_file_names=None, num_keypoints=9, max_num_gt=50, cam_params=None, corners3D=None):

        # root             : list of training or test images
        # shape            : shape of the image input to the network
        # shuffle          : whether to shuffle or not 
        # tranform         : any pytorch-specific transformation to the input image 
        # target_transform : any pytorch-specific tranformation to the target output
        # train            : whether it is training data or test data
        # seen             : the number of visited examples (iteration of the batch x batch size) # TODO: check if this is correctly assigned
        # batch_size       : how many examples there are in the batch
        # num_workers      : check what this is
        # bg_file_names    : the filenames for images from which you assign random backgrounds

        # read the the list of dataset images
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if cam_params is not None:
            # reject any gt's that are not in FOV!! (or are too far away)
            keep_list = []
            box_gt = []
            def truths_length(truths, max_num_gt=50):
                for i in range(max_num_gt):
                    if truths[i][1] == 0:
                        return i
            my_camera = camera(*cam_params)  # cam_params = (K, dist_coefs, im_width, im_height, tf_cam_ego)
            for l in self.lines:
                # Get the image path
                num_img = len(self.lines)
                imgpath = l.rstrip()
                labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
                num_labels = 2*num_keypoints+3 # +2 for ground-truth of width/height , +1 for class label
                label = torch.zeros(max_num_gt*num_labels)
                if os.path.getsize(labpath):
                    ow = my_camera.im_w
                    oh = my_camera.im_h
                    tmp = torch.from_numpy(read_truths_args(labpath))
                    tmp = tmp.view(-1)
                    tsz = tmp.numel()
                    if tsz > max_num_gt*num_labels:
                        label = tmp[0:max_num_gt*num_labels]
                    elif tsz > 0:
                        label[0:tsz] = tmp
                label = label.cuda()
                truths = label.view(-1, num_labels)
                num_gts = truths_length(truths)
                for k in range(num_gts):
                    box_gt = list()
                    for j in range(1, 2*num_keypoints+1):
                        box_gt.append(truths[k][j])
                    box_gt.extend([1.0, 1.0])
                    box_gt.append(truths[k][0])
                corners2D_gt = np.array(np.reshape(box_gt[:18], [-1, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * my_camera.im_w
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * my_camera.im_h

                R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'), corners2D_gt, np.array(my_camera.K, dtype='float32'))
                MAX_DIST_TO_KEEP = 4
                if np.any(corners2D_gt < 0) or np.any(corners2D_gt[:, 0] > my_camera.im_w) or np.any(corners2D_gt[:, 1] > my_camera.im_h) or la.norm(t_gt) > MAX_DIST_TO_KEEP:
                    keep_list.append(False)
                else:
                    keep_list.append(True)
                
            self.lines = [l for (l, my_bool) in zip(self.lines, keep_list) if my_bool]
            if len(self.lines) == 0:
                raise RuntimeError("ALL IMAGES SCREENED OUT (out of original {} images)".format(num_img))
            print('Keeping {} / {} images (throwing out images with no drone in FOV)'.format(len(self.lines), num_img))

        # Shuffle
        if shuffle:
            random.shuffle(self.lines)

        # Initialize variables
        self.nSamples         = len(self.lines)
        self.transform        = transform
        self.target_transform = target_transform
        self.train            = train
        self.shape            = shape
        self.seen             = seen
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.bg_file_names    = bg_file_names
        self.cell_size        = cell_size
        self.nbatches         = self.nSamples // self.batch_size
        self.num_keypoints    = num_keypoints
        self.max_num_gt       = max_num_gt # maximum number of ground-truth labels an image can have

    # Get the number of samples in the dataset
    def __len__(self):
        return self.nSamples

    # Get a sample from the dataset
    def __getitem__(self, index):

        # Ensure the index is smallet than the number of samples in the dataset, otherwise return error
        assert index <= len(self), 'index range error'

        # Get the image path
        imgpath = self.lines[index].rstrip()

        # Decide which size you are going to resize the image depending on the epoch (10, 20, etc.)
        if self.train and index % self.batch_size== 0:
            if self.seen < 10*self.nbatches*self.batch_size:
               width = 13*self.cell_size
               self.shape = (width, width)
            elif self.seen < 20*self.nbatches*self.batch_size:
               width = (random.randint(0,7) + 13)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 30*self.nbatches*self.batch_size:
               width = (random.randint(0,9) + 12)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 40*self.nbatches*self.batch_size:
               width = (random.randint(0,11) + 11)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 50*self.nbatches*self.batch_size:
               width = (random.randint(0,13) + 10)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 60*self.nbatches*self.batch_size:
               width = (random.randint(0,15) + 9)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 70*self.nbatches*self.batch_size:
               width = (random.randint(0,17) + 8)*self.cell_size
               self.shape = (width, width)
            else: 
               width = (random.randint(0,19) + 7)*self.cell_size
               self.shape = (width, width)

        if self.train:
            # Decide on how much data augmentation you are going to apply
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            # Get background image path
            random_bg_index = random.randint(0, len(self.bg_file_names) - 1)
            bgpath = self.bg_file_names[random_bg_index]    

            # Get the data augmented image and their corresponding labels
            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure, bgpath, self.num_keypoints, self.max_num_gt)

            # Convert the labels to PyTorch variables
            label = torch.from_numpy(label)
        
        else:
            # Get the validation image, resize it to the network input size
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            # Read the validation labels, allow upto 50 ground-truth objects in an image
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            num_labels = 2*self.num_keypoints+3 # +2 for ground-truth of width/height , +1 for class label
            label = torch.zeros(self.max_num_gt*num_labels)
            if os.path.getsize(labpath):
                ow, oh = img.size
                tmp = torch.from_numpy(read_truths_args(labpath))
                tmp = tmp.view(-1)
                tsz = tmp.numel()
                if tsz > self.max_num_gt*num_labels:
                    label = tmp[0:self.max_num_gt*num_labels]
                elif tsz > 0:
                    label[0:tsz] = tmp

        # Tranform the image data to PyTorch tensors
        if self.transform is not None:
            img = self.transform(img)

        # If there is any PyTorch-specific transformation, transform the label data
        if self.target_transform is not None:
            label = self.target_transform(label)

        # Increase the number of seen examples
        self.seen = self.seen + self.num_workers

        # Return the retrieved image and its corresponding label
        return (img, label)
