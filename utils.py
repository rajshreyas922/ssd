import numpy as np
import cv2
import numpy as np
from dataset import iou
import sys
np.set_printoptions(threshold=sys.maxsize)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes



def visualize_pred_train(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, og_box):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                # Ground truth bounding box (image1)
    
                for b in og_box:
                    x_min, y_min, x_max, y_max = b
                    x_min = int(x_min*320)
                    y_min = int(y_min*320)
                    x_max = int(x_max*320)
                    y_max = int(y_max*320)
                    cv2.rectangle(image1, (x_min, y_min), (x_max, y_max), colors[j], 2)

                # Ground truth default box (image2)
                x_center, y_center, box_width, box_height = boxs_default[i, :4]
                x_min = int((x_center - box_width / 2) * image.shape[1])
                y_min = int((y_center - box_height / 2) * image.shape[0])
                x_max = int((x_center + box_width / 2) * image.shape[1])
                y_max = int((y_center + box_height / 2) * image.shape[0])
                cv2.rectangle(image2, (x_min, y_min), (x_max, y_max), colors[j], 2)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                # Predicted bounding box (image3)
                dx, dy, dw, dh = pred_box[i]
                x_center, y_center, box_width, box_height = boxs_default[i, :4]
                x_center = (dx * box_width) + x_center
                y_center = (dy * box_height) + y_center
                box_width = np.exp(dw) * box_width
                box_height = np.exp(dh) * box_height
                x_min = int((x_center - box_width / 2) * image.shape[1])
                y_min = int((y_center - box_height / 2) * image.shape[0])
                x_max = int((x_center + box_width / 2) * image.shape[1])
                y_max = int((y_center + box_height / 2) * image.shape[0])
     
                cv2.rectangle(image3, (x_min, y_min), (x_max, y_max), colors[j], 2)

                # Predicted default box (image4)
                x_center, y_center, box_width, box_height = boxs_default[i, :4]
                x_min = int((x_center - box_width / 2) * image.shape[1])
                y_min = int((y_center - box_height / 2) * image.shape[0])
                x_max = int((x_center + box_width / 2) * image.shape[1])
                y_max = int((y_center + box_height / 2) * image.shape[0])
                cv2.rectangle(image4, (x_min, y_min), (x_max, y_max), colors[j], 2)

    # Combine and display images...
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.

def visualize_pred_test(windowname, pred_confidence, pred_box, ann_confidence, ann_box, nms_box, nms_conf, image_, boxs_default, og_box):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                # Ground truth bounding box (image1)
    
                for b in og_box:
                    
                    x_min, y_min, x_max, y_max = b
                    x_min = int(x_min*320)
                    y_min = int(y_min*320)
                    x_max = int(x_max*320)
                    y_max = int(y_max*320)
                    
                    cv2.rectangle(image1, (x_min, y_min), (x_max, y_max), colors[j], 2)

                # Ground truth default box (image2)
                x_center, y_center, box_width, box_height = boxs_default[i, :4]
                x_min = int((x_center - box_width / 2) * image.shape[1])
                y_min = int((y_center - box_height / 2) * image.shape[0])
                x_max = int((x_center + box_width / 2) * image.shape[1])
                y_max = int((y_center + box_height / 2) * image.shape[0])
                cv2.rectangle(image2, (x_min, y_min), (x_max, y_max), colors[j], 2)
    

    for i in range(len(nms_conf)):
        x_min, y_min, x_max, y_max = nms_box[i]
        cat = np.argmax(nms_conf[i])
        cv2.rectangle(image3, (x_min, y_min), (x_max, y_max), colors[cat], 2)

    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                # Predicted bounding box (image3)
                
                

                # Predicted default box (image4)
                x_center, y_center, box_width, box_height = boxs_default[i, :4]
                x_min = int((x_center - box_width / 2) * image.shape[1])
                y_min = int((y_center - box_height / 2) * image.shape[0])
                x_max = int((x_center + box_width / 2) * image.shape[1])
                y_max = int((y_center + box_height / 2) * image.shape[0])
                cv2.rectangle(image4, (x_min, y_min), (x_max, y_max), colors[j], 2)

    # Combine and display images...
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.




def calculate_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes.
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def convert_to_actual_boxes(box_, boxs_default, image_shape):
    actual_boxes = []
    for i in range(len(box_)):
        dx, dy, dw, dh = box_[i]
        x_center, y_center, box_width, box_height = boxs_default[i, :4]
        x_center = (dx * box_width) + x_center
        y_center = (dy * box_height) + y_center
        box_width = np.exp(dw) * box_width
        box_height = np.exp(dh) * box_height

        x_min = int((x_center - box_width / 2) * image_shape[1])
        y_min = int((y_center - box_height / 2) * image_shape[0])
        x_max = int((x_center + box_width / 2) * image_shape[1])
        y_max = int((y_center + box_height / 2) * image_shape[0])

        actual_boxes.append([x_min, y_min, x_max, y_max])
    return np.array(actual_boxes)



def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area != 0 else 0


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.1, threshold=0.5):

    confidence_ = confidence_[:, 0:3]

    actual_boxes = convert_to_actual_boxes(box_, boxs_default, (320, 320))
    selected_boxes = []  # store the selected bounding boxes
    selected_classes_and_confidences = []  # To store tuples of (class_index, confidence)
    
    while len(actual_boxes) > 0:
        # Find the box with the highest confidence across the first three classes
        highest_prob_index = np.argmax(confidence_.max(axis=1))
        highest_class_index = np.argmax(confidence_[highest_prob_index]) 
        highest_prob = np.max(confidence_[highest_prob_index])
        
        # Check if the highest probability is greater than the threshold
        if highest_prob <= threshold:
            break  
        
        # Move the box with the highest probability to the selected list
        x = actual_boxes[highest_prob_index]
        selected_boxes.append(x)
        selected_classes_and_confidences.append((highest_class_index, highest_prob))
        
        # Remove the selected box
        actual_boxes = np.delete(actual_boxes, highest_prob_index, axis=0)
        confidence_ = np.delete(confidence_, highest_prob_index, axis=0)
        
        # Remove boxes with IOU greater than the overlap threshold with x
        ious = np.array([iou(x, box) for box in actual_boxes])
        indices_to_keep = np.where(ious <= overlap)[0]
        
        actual_boxes = actual_boxes[indices_to_keep]
        confidence_ = confidence_[indices_to_keep]

    confidences = create_confidence_array(selected_classes_and_confidences)

    return np.array(confidences), np.array(selected_boxes)

def create_confidence_array(selected_classes_and_confidences):
    # Initialize an array of zeros with shape (n, 4), assuming there are 3 classes (0-2 indices) + 1 for potential extra class
    # Adjust the number of columns (4 in this case) if you have more classes
    n = len(selected_classes_and_confidences)
    confidences_array = np.zeros((n, 4))
    
    # Fill in the confidence values at the appropriate class indices
    for i, (class_index, confidence) in enumerate(selected_classes_and_confidences):
        confidences_array[i, class_index] = confidence
    
    return confidences_array
    



def extract_class_numbers(pred_confidence):
    # Convert to a numpy array if it's not already one
    pred_confidence = np.array(pred_confidence)
    # Find the index of the maximum confidence score in each row
    class_numbers = np.argmax(pred_confidence, axis=1)
    return class_numbers.tolist()


def update_precision_recall(pred_conf, pred_boxes, true_conf, true_boxes, default_boxes, precision_list, recall_list, og_boxes):
    iou_threshold = 0.5
    TP = 0
    FP = 0
    FN = 0

    matched_ground_truths = [False] * len(true_boxes)

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(true_boxes):
            iou_ = iou(pred_box, gt_box)
            if iou_ > best_iou:
                best_iou = iou_
                best_gt_index = gt_index
        
        if best_iou >= iou_threshold:
            if not matched_ground_truths[best_gt_index]:
                TP += 1
                matched_ground_truths[best_gt_index] = True
            else:
                FP += 1
        else:
            FP += 1

    FN = len(true_boxes) - sum(matched_ground_truths)

    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    precision_list.append(precision)
    recall_list.append(recall)



    sorted_recall = [0] + sorted_recall
    sorted_precision = [1] + sorted_precision

    ap = np.trapz(sorted_precision, sorted_recall)
    return ap








