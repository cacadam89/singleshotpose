import numpy as np 
import matplotlib.pyplot as plt
import pdb
import PIL
import PIL.Image as Image
import cv2
# from natsort import natsorted

from torchvision import datasets, transforms
from torch.autograd import Variable # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html

import dataset
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from utils import *


# First Plot the losses
data = np.load("./backup/mslquad/costs.npz")
losses = data['training_losses']
N = len(losses)
plt.plot(range(N), losses, 'b-')
ax = plt.gca()
ax.set_title("Loss")
ax.axes.set_xlabel("itr... epoch??")
ax.axes.set_ylabel("loss")
plt.show(block=False)
###########################################

datacfg = 'cfg/mslquad.data'
modelcfg = 'cfg/yolo-pose.cfg'

# Get the dataloader for test data
data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(modelcfg)[0]
testlist      = data_options['valid']
test_width  = int(net_options['test_width'])
test_height = int(net_options['test_height'])
num_workers   = int(data_options['num_workers'])

# Specify which gpus to use
use_cuda      = True
seed          = int(time.time())
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

# Specifiy the model and the loss
model       = Darknet(modelcfg)
region_loss = RegionLoss(num_keypoints=9, num_classes=1, anchors=[], num_anchors=1, pretrain_num_epochs=15)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}


test_loader = torch.utils.data.DataLoader(dataset.listDataset(testlist, 
                                                             shape=(test_width, test_height),
                                                             shuffle=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),]), 
                                                             train=False),
                                            batch_size=1, shuffle=False, **kwargs)

pdb.set_trace()
for batch_idx, (data, target) in enumerate(test_loader):
    t1 = time.time()
    # Pass the data to GPU
    if use_cuda:
        data = data.cuda()
        target = target.cuda()
    # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
    data = Variable(data, volatile=True)
    t2 = time.time()
    # Formward pass
    output = model(data).data  
    t3 = time.time()
    # Using confidence threshold, eliminate low-confidence predictions
    all_boxes = get_region_boxes(output, num_classes, num_keypoints)        
    t4 = time.time()
    # Iterate through all batch elements
    for box_pr, target in zip([all_boxes], [target[0]]):
        # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
        truths = target.view(-1, num_keypoints*2+3)
        # Get how many objects are present in the scene
        num_gts    = truths_length(truths)
        # Iterate through each ground-truth object
        for k in range(num_gts):
            box_gt = list()
            for j in range(1, 2*num_keypoints+1):
                box_gt.append(truths[k][j])
            box_gt.extend([1.0, 1.0])
            box_gt.append(truths[k][0])
                
            # Denormalize the corner predictions 
            corners2D_gt = np.array(np.reshape(box_gt[:num_keypoints*2], [num_keypoints, 2]), dtype='float32')
            corners2D_pr = np.array(np.reshape(box_pr[:num_keypoints*2], [num_keypoints, 2]), dtype='float32')
            corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
            corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height               
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

            # Compute corner prediction error
            corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
            corner_dist = np.mean(corner_norm)
            errs_corner2D.append(corner_dist)

            # Compute [R|t] by pnp
            R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
            R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

            # Compute errors
            # Compute translation error
            trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
            errs_trans.append(trans_dist)

            # Compute angle error
            angle_dist   = calcAngularDistance(R_gt, R_pr)
            errs_angle.append(angle_dist)

            # Compute pixel error
            Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
            Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
            proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration) 
            proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
            norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
            pixel_dist   = np.mean(norm)
            errs_2d.append(pixel_dist)

            # Compute 3D distances
            transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
            transform_3d_pred = compute_transformation(vertices, Rt_pr)  
            norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
            vertex_dist       = np.mean(norm3d)    
            errs_3d.append(vertex_dist)  

            # Sum errors
            testing_error_trans  += trans_dist
            testing_error_angle  += angle_dist
            testing_error_pixel  += pixel_dist
            testing_samples      += 1

    t5 = time.time()


input("\n----------------  Hit enter to quit  ----------------n")