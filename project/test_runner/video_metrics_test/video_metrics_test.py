import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from torchvision.transforms import Resize

from video_object_detection_metrics.center_position_error import center_position_error
from video_object_detection_metrics.scale_and_retio_error import scale_and_ratio_error


class ImageVector:
    
    def __init__(self,
                 img_path: str,
                 max_shift: int,
                 stride: int=1):
        
        self.max_shift = max_shift
        self.stride = stride
        self.img_path = img_path
        self.img = self.load_image(img_path)
        self.img_vector = self.create_shifted_imgs(self.img, max_shift,stride)
    
    def create_shifted_imgs(self,img, max_shift, stride):
        #img format needs to be (H,W,C), values in range [0,1]
        #max_shift is the number of output images
        img_width = img.shape[1]
        img_height = img.shape[0]
        
        new_width = img_width + (max_shift*stride)
        new_height = img_height + (max_shift*stride)
        
        resize_for_crop = Resize((new_height, new_width))
        resized_tensor = resize_for_crop(torch.tensor(img).permute(2, 0, 1)) 
        
        shifted_imgs = []
        for i in range(max_shift+1):
            for j in range(max_shift+1):
                shift_x = i * stride
                shift_y = j * stride
                # print(f"shift x: {shift_x}")
                # print(f"shift y: {shift_y}")
                shifted_img = resized_tensor[:, shift_y:shift_y+img_height, shift_x:shift_x+img_width]
                shifted_imgs.append(shifted_img.permute(1, 2, 0).numpy())
        
        return shifted_imgs
    
    def load_image(self, img_path):
        with Image.open(img_path) as img:
            img = np.array(img)
            print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


class VideoMetricsTest:
    #TODO: where should we put the threshold? in process predictions/in gt/ test_error
    
    def __init__(self,
                 image_vector: ImageVector,
                 predictions,
                 score_threshold=0.7,
                 iou_threshold=0.9):
        self.score_threshold=score_threshold
        self.iou_threshold = iou_threshold
        self.image_vector = image_vector
        
        self.predictions = self.process_predictions(predictions=predictions,
                                                    score_threshold=score_threshold)
        
        self.gt_vector = self.create_gt()

    
    def process_predictions(self,
                            predictions:list,
                            score_threshold):
        processed_predictions = []
        self.labels = []
        for result in predictions:
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                bboxes = np.vstack(bbox_result)

                bboxes[:, [0,1,2,3,4]] = bboxes[:, [4,0,1,2,3]]
                
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                if score_threshold > 0:
                    assert bboxes is not None and bboxes.shape[1] == 5
                    scores = bboxes[:, 0]
                    inds = scores > score_threshold
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
            self.labels.append(labels)
            processed_predictions.append(bboxes)
        return processed_predictions
    
    def create_gt(self):
        first_frame_gt = self.predictions[0]
        first_frame_gt[:,0] = range(len(first_frame_gt))
        gt_vector = []
        max_shift = self.image_vector.max_shift
        stride = self.image_vector.stride
        for i in range(max_shift+1):
            for j in range(max_shift+1):
                shift_x = i * stride
                shift_y = j * stride
                gt_vector_frame = [[idx, max(0, x1 - shift_x), max(0,y1 - shift_y), max(0,x2- shift_x), max(0,y2 - shift_y)] for [idx, x1, y1, x2, y2] in first_frame_gt]
                gt_vector.append(gt_vector_frame)
        return gt_vector
    
    def test_center_position_error(self):
        return center_position_error(video_predicts=self.predictions,
                                     video_ground_truths=self.gt_vector,
                                     predict_score_threshold=self.score_threshold,
                                     iou_threshold=self.iou_threshold)

    def test_scale_and_ratio_error(self):
        return scale_and_ratio_error(video_predicts=self.predictions,
                                     video_ground_truths=self.gt_vector,
                                     predict_score_threshold=self.score_threshold,
                                     iou_threshold=self.iou_threshold)
