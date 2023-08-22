import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm


def iou(box1,box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    #Area of overalp
    w1 = tf.abs(x1_max - x2_min)
    h1 = tf.abs(y1_max - y2_max)
    area_overlap = w1 * h1
    area_of_union = (y2_max-y2_min) * (x2_max - x2_min)
    
    epsilon= 1e7
    iou_value = area_overlap/(area_of_union + epsilon)
    return iou_value
    

def non_max_suppression(boxes, iou_threshold):
    selected_boxes = []
    
    # Sort the boxes by their confidence score (if available)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    while len(boxes) > 0:
        best_box = boxes[0]
        selected_boxes.append(best_box)
        boxes = [box for box in boxes[1:] if iou(best_box, box) < iou_threshold]
    
    return selected_boxes


def convert_cells_to_bboxes(predictions,anchors,s,is_predictions=True):
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[...,1:5]
    
    if is_predictions:
         
        #x,y predicted center coordinates of the bounding box
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        #Pass x and y to sigmoid 
        box_predictions[..., 0:2] = tf.sigmoid(box_predictions[..., 0:2])
        #pass height and width to exp
        box_predictions[..., 2:] = tf.exp(
            box_predictions[..., 2:]) * anchors
        scores = tf.sigmoid(predictions[..., 0:1])
        best_class = tf.expand_dims(tf.argmax(predictions[..., 5:], axis=1),axis=1)
        
    else:
        scores = predictions[..., 0:1]
        class_scores = predictions[..., 5:]
        best_class = tf.expand_dims(tf.argmax(class_scores,axis=1),axis=1)
        
    #Create cell
    cell_indices = (np.arange(s).repeat(predictions.shape[0],3,s,1))
    # Calculate x, y, width and height with proper scaling