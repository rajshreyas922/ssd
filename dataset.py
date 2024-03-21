import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#generate default bounding boxes
def default_box_generator(layers = [10,5,3,1], large_scale = [0.2,0.4,0.6,0.8], small_scale = [0.1,0.3,0.5,0.7]):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]


    total_cells = sum([layer_size**2 for layer_size in layers])
    total_boxes = total_cells * 4
    boxes = np.zeros((total_boxes, 8))

    idx = 0
    for i in range(len(layers)):
        cell_size = 1.0 / layers[i]
        for y in range(layers[i]):
            for x in range(layers[i]):
                center_x = (x + 0.5) * cell_size
                center_y = (y + 0.5) * cell_size

                for scale in [(small_scale[i], small_scale[i]), 
                              (large_scale[i], large_scale[i]), 
                              (large_scale[i] * np.sqrt(2), large_scale[i] / np.sqrt(2)), 
                              (large_scale[i] / np.sqrt(2), large_scale[i] * np.sqrt(2))]:
                    box_width = scale[0]
                    box_height = scale[1]
                    x_min = center_x - box_width / 2
                    y_min = center_y - box_height / 2
                    x_max = center_x + box_width / 2
                    y_max = center_y + box_height / 2

                    boxes[idx] = [center_x, center_y, box_width, box_height, x_min, y_min, x_max, y_max]
                    idx += 1

    return boxes



#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    # ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    if cat_id == 0:
        one_hot_label = np.array([1, 0, 0, 0])
    elif cat_id == 1:
        one_hot_label = np.array([0, 1, 0, 0])
    elif cat_id == 2:
        one_hot_label = np.array([0, 0, 1, 0])

    gt_width = x_max - x_min
    gt_height = y_max - y_min
    gt_box = [x_min + gt_width / 2.0, 
              y_min + gt_height / 2.0, 
              gt_width, 
              gt_height]

    ious_true = ious > threshold
    max_iou_idx = np.argmax(ious)
    ious_true[max_iou_idx] = True
    def_x_centre = boxs_default[ious_true, 0]
    def_y_centre = boxs_default[ious_true, 1]
    def_width = boxs_default[ious_true, 2]
    def_height = boxs_default[ious_true, 3]
    ann_box[ious_true, :] = np.column_stack((
        (gt_box[0] - def_x_centre) / def_width, 
        (gt_box[1] - def_y_centre) / def_height,
        np.log(gt_box[2] / def_width), 
        np.log(gt_box[3] / def_height)
    ))

    
    ann_confidence[ious_true, :] = one_hot_label
    
    # ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    
    # This part is done above


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        image = cv2.imread(img_name)
        height, width, _ = image.shape
        x_scale = 1 / width
        y_scale = 1 / height
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32())
        
        with open(ann_name, 'r') as fobj:
            og_box = np.zeros((5,4))
            i = 0
            for line in fobj:
                class_id, x_min_og, y_min_og, box_width, box_height = [float(num) for num in line.split()]
                class_id = int(class_id)
                x_min = (x_min_og) * x_scale
                y_min = (y_min_og) * y_scale
                x_max = (x_min_og + box_width) * x_scale
                y_max = (y_min_og + box_height) * y_scale
                box_coordinates = [x_min, y_min, x_max, y_max]
                og_box[i] = (box_coordinates)
                i += 1


                # image1 = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), [255,0,0],2)
                # cv2.imshow("dateset",image1)
                # cv2.waitKey(0)
                # Update ann_box and ann_confidence using the match function
                
                match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min, y_min, x_max, y_max)

        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)      
        return image, ann_box, ann_confidence, np.array(og_box)

# imgdir = 'data/train/images/'  # Replace with the path to your image directory
# anndir = 'data/train/annotations/'  # Replace with the path to your annotation directory
# class_num = 4  # Number of classes (including background)
# boxs_default = default_box_generator()  # Example default boxes
# coco_dataset = COCO(imgdir, anndir, class_num, boxs_default)
# #68: cat
# #5406: two humans

# index = 5406  # Index of the item to retrieve
# image, ann_box, ann_confidence, _ = coco_dataset.__getitem__(index)

# def draw_boxes(image, ann_box, boxs_default, ann_confidence):
#     # Define colors for different classes
#     colors = {0: 'blue',  # Cat
#               1: 'green', # Dog
#               2: 'red'}   # Person

#     fig, ax = plt.subplots(1)
#     ax.imshow(image)

#     for i in range(ann_box.shape[0]):
#         if np.max(ann_box[i, :]) != 0:  # Filter out background boxes
#             dx, dy, dw, dh = ann_box[i, :]
#             x_center, y_center, box_width, box_height = boxs_default[i, :4]

#             # Convert back to image coordinates
#             x_center = (dx * box_width) + x_center
#             y_center = (dy * box_height) + y_center
#             box_width = np.exp(dw) * box_width
#             box_height = np.exp(dh) * box_height

#             # Convert to x_min, y_min, x_max, y_max
#             x_min = (x_center - box_width / 2.0) * image.shape[1]
#             y_min = (y_center - box_height / 2.0) * image.shape[0]
#             box_width *= image.shape[1]
#             box_height *= image.shape[0]

#             # Determine the class and choose color
#             class_id = np.argmax(ann_confidence[i, :-1])
#             color = colors.get(class_id, 'yellow')  

#             # Draw the box
#             rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=2, edgecolor=color, facecolor='none')
#             ax.add_patch(rect)

#     plt.show()

# # Convert image to RGB and draw boxes
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# draw_boxes(image_rgb, ann_box, boxs_default, ann_confidence)


