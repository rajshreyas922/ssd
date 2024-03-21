import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

    [batch_size, num_of_boxes, num_of_classes] = list(ann_confidence.shape)
    pred_confidence = torch.reshape(pred_confidence, (batch_size * num_of_boxes, num_of_classes))
    ann_confidence = torch.reshape(ann_confidence, (batch_size * num_of_boxes, num_of_classes))
    pred_box = torch.reshape(pred_box, (batch_size * num_of_boxes, 4))
    ann_box = torch.reshape(ann_box, (batch_size * num_of_boxes, 4))

    #If the confidence of background is 0, the box must contain an object.
    is_obj = ann_confidence[:, -1] == 0  #1 in the index where a box contains an object.
    noobj = ~is_obj  #1 in the index where a box does not contain an object (Background).

    cls_loss = (F.binary_cross_entropy(pred_confidence[is_obj, :], ann_confidence[is_obj, :], reduction='mean') +
                3 * F.binary_cross_entropy(pred_confidence[noobj, :], ann_confidence[noobj, :], reduction='mean'))
    box_loss = F.smooth_l1_loss(pred_box[is_obj, :], ann_box[is_obj, :], reduction='mean')

    loss = cls_loss + box_loss

    return loss



class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.before_split = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # conv2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # conv3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # conv4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # conv5
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # conv6
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # conv7
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # conv8
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # conv9
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # conv10
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # conv11
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # conv12
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # conv13
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv_box1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv_confidence1 = nn.Conv2d(256, self.class_num * 4, kernel_size=3, stride=1, padding=1)

        self.down2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv_box2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv_confidence2 = nn.Conv2d(256, self.class_num * 4, kernel_size=3, stride=1, padding=1)

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv_box3 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv_confidence3 = nn.Conv2d(256, self.class_num * 4, kernel_size=3, stride=1, padding=1)

        self.conv_box4 = nn.Conv2d(256, 16, kernel_size=1, stride=1)
        self.conv_confidence4 = nn.Conv2d(256, self.class_num * 4, kernel_size=1, stride=1)
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        x = self.before_split(x)
        box1 = self.conv_box1(x)
        box1 = torch.reshape(box1, (-1, 16, 100))
        confidence1 = self.conv_confidence1(x)
        confidence1 = torch.reshape(confidence1, (-1, self.class_num * 4, 100))

        x = self.down1(x)
        box2 = self.conv_box1(x)
        box2 = torch.reshape(box2, (-1, 16, 25))
        confidence2 = self.conv_confidence1(x)
        confidence2 = torch.reshape(confidence2, (-1, self.class_num * 4, 25))

        x = self.down2(x)
        box3 = self.conv_box1(x)
        box3 = torch.reshape(box3, (-1, 16, 9))
        confidence3 = self.conv_confidence1(x)
        confidence3 = torch.reshape(confidence3, (-1, self.class_num * 4, 9))

        x = self.down3(x)
        box4 = self.conv_box1(x)
        box4 = torch.reshape(box4, (-1, 16, 1))
        confidence4 = self.conv_confidence1(x)
        confidence4 = torch.reshape(confidence4, (-1, self.class_num * 4, 1))

        bboxes = torch.cat((box1, box2, box3, box4), 2)
        confidence = torch.cat((confidence1, confidence2, confidence3, confidence4), 2)

        bboxes = bboxes.permute(0, 2, 1)
        confidence = confidence.permute(0, 2, 1)

        bboxes = torch.reshape(bboxes, (-1, 540, 4))
        confidence = torch.reshape(confidence, (-1, 540, self.class_num))

        confidence = F.softmax(confidence, dim=2)

        return confidence,bboxes



