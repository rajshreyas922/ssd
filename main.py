import argparse
import os
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import random_split
from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--output', type=str, help='Output file path')  # Add this line for the --output argument

args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test

class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test and not args.output == "t":
    
    full_dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)

    train_size = int(0.9 * len(full_dataset))
    validation_size = len(full_dataset) - train_size

    # Splitting the dataset
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    # Creating data loaders for train and validation
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(network.parameters(), lr = 1e-4, weight_decay=1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, og_box_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()
            # print(og_box_.shape)
            # print(type(og_box_))
            # print(type(og_box_[0]))
            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred_train("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, og_box_[0].numpy())

        #VALIDATION
        network.eval()
        precision_ = []
        recall_ = []
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        for i, data in enumerate(dataloader_validation, 0):
            images_, ann_box_, ann_confidence_, og_box_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()
 
            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(nms_conf, nms_box, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_, og_box_)

        #map_value = generate_mAP(precision_, recall_)
        #print(f"mAP: {map_value}")
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred_train("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, og_box_[0].numpy())
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        if epoch%10==9:
        #save weights
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')


elif args.test:
    #TEST
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network_with_regularization.pth'))
    network.eval()
    
    precision_ = []
    recall_ = []
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, og_box_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()


        # img = cv2.imread('data/iris cat/unnamed(21).jpg')
        # img = cv2.resize(img, (320, 320))
        # img = np.transpose(img, (2, 0, 1)) 
        # img = torch.from_numpy(img)
        # img = img.to('cuda')
        # img = img.unsqueeze(0)  
        # print(img.shape)

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        nms_conf,nms_box = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        #optional: implement a function to accumulate precision and recall to compute mAP or F1.
        #update_precision_recall(nms_conf, nms_box, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_, og_box_)
                    
   
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        
        visualize_pred_test("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), nms_box, nms_conf, images_[0].numpy(), boxs_default, og_box_[0].numpy())
        cv2.waitKey(0)
        
 
    p_avg = np.average(precision_)
    r_avg = np.average(recall_)
    print((2*p_avg*r_avg)/(p_avg+r_avg))


else:
    network.load_state_dict(torch.load('network_with_regularization.pth'))
    network.eval()
    test_path = 'data/test/images'
    file_names = os.listdir(test_path)

    precision_ = []
    recall_ = []
    for i in file_names:
        img = cv2.imread(os.path.join(test_path, i))
        img = cv2.resize(img, (320, 320))
        output = img.copy()  # Make a copy of the image for drawing

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()  # Ensure img is float before unsqueeze
        img = img.to('cuda')
        img = img.unsqueeze(0)

        pred_confidence, pred_box = network(img)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        nms_conf, nms_box = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
        # Your optional code for precision and recall

        for i in range(len(nms_conf)):
            x_min, y_min, x_max, y_max = nms_box[i]
            cat = np.argmax(nms_conf[i])
            cv2.rectangle(output, (x_min, y_min), (x_max, y_max), colors[cat], 2)

        # Display or save the output image
        cv2.imshow('Detection', output)
        cv2.waitKey(0)