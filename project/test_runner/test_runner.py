import os
import sys
import csv
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
import json
from pycocotools.coco import COCO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from video_object_detection_metrics.center_position_error import center_position_error
from video_object_detection_metrics.scale_and_retio_error import scale_and_ratio_error
from video_object_detection_metrics.fragment_error import fragment_error

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import torch, torchvision
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

from test_runner_utils import resize_image_and_bboxes, create_shifted_imgs, format_predictions, format_ground_truths, handle_cpe_error_dict, handle_sre_error_dict, write_to_csv

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Evaluation Script")
    parser.add_argument('--is_afc',
                        dest="is_afc",
                        action='store_true',
                        default=False,
                        help="Use AFC configuration if set to true.")
    parser.add_argument('--max_shift',
                        dest="max_shift",
                        type=int,
                        required=True,
                        help="Maximum shift value for creating shifted images.")
    parser.add_argument('--stride',
                        dest="stride",
                        type=int,
                        default=1,
                        help="Stride value for creating shifted images.")
    parser.add_argument('-r', '--random_weights',
                        dest="random_weights",
                        action="store_true",
                        default=False,
                        help="use random weights")
    parser.add_argument('-fpn', 
                        dest="is_afc_fpn",
                        action="store_true",
                        default=False,
                        help="use afc fpn config")
    parser.add_argument('-is_cyclic', 
                        dest="is_cyclic",
                        action="store_true",
                        default=True,
                        help="use cyclic shift or crop shift")
    parser.add_argument('--start', 
                        dest="start_index",
                        type=int,
                        default=0,
                        help="index to start running from")
    parser.add_argument('-gpu', 
                        dest="gpu_num",
                        type=int,
                        default=0,
                        help="gpu number")
    return parser.parse_args()
    
def test_runner(is_afc: bool,
                max_shift: int,
                stride: int=1,
                random_weights: bool=False,
                cyclic_flag: bool=False,
                start_index: int=0,
                end_index: int=None,
                is_afc_fpn: bool=False,
                gpu_num: int=0,
                csv_filename: str=None,
                is_only_fpn: bool=False,):
    # change the path of the config files and the checkpoint to the correct path on your computer
    if is_afc:
        # Choose to use a config and initialize the detector
        if is_afc_fpn:
            config="/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_convnext_afc_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/mask_rcnn_nir_hadar_convnext_afc_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py"
            checkpoint = '/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_convnext_afc_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/latest.pth'
        
        elif is_only_fpn:
            config="/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_convnext_afc_only_fpn_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/mask_rcnn_nir_hadar_convnext_afc_only_fpn_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py"
            checkpoint = "/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_convnext_afc_only_fpn_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/latest.pth"
            
        else: ##AFC Backbone
            config = '/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_convnext_afc_only_backbone_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/mask_rcnn_nir_hadar_convnext_afc_only_backbone_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py'
            checkpoint = '/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_convnext_afc_only_backbone_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/latest.pth'
    #baseline
    else:
        config = '/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_baseline_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/mask_rcnn_nir_hadar_baseline_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py'
        checkpoint = '/home/nir_hadar/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_nir_hadar_baseline_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/latest.pth'
    
    if random_weights:
        checkpoint=None
        
    torch.cuda.empty_cache()
    # initialize the detector
    model = init_detector(config, checkpoint, device=f'cuda:{gpu_num}')
    
    dataDir = '/home/data/coco'  # Adjust this to your actual dataset path
    dataType = 'val2017'  # Only process validation data
    annFile = os.path.join(dataDir, 'annotations', f'instances_{dataType}.json')
    imageDir = os.path.join(dataDir, dataType)
    
    ground_truth_dict = {}
    # Initialize COCO API
    coco = COCO(annFile)
    
    imgIds = coco.getImgIds()

    for imgId in imgIds:
        # Load image information
        img = coco.loadImgs(imgId)[0]
        image_path = os.path.join(imageDir, img['file_name'])
        
        # Load annotations for the image
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # Collect ground truth data
        ground_truths = []
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            ground_truths.append({
                'bbox': bbox,
                'category_id': category_id,
                'category_name': category_name
            })
        ground_truth_dict[img['file_name']] = ground_truths
    
    images_file_names = [file for file in os.listdir(os.path.join(dataDir, dataType)) if os.path.isfile(os.path.join(os.path.join(dataDir, dataType), file))]
    
   
    if end_index is None:
        end_index = len(images_file_names)
    if csv_filename is None:
        csv_filename = f"max_shift_{max_shift}_stride_{stride}"
    
        if is_afc:
            csv_filename += "_afc"
        if random_weights:
            csv_filename += "_random_weights"
        if cyclic_flag:
            csv_filename += f"_cyclic.csv"
        else:
            csv_filename += f"_crop.csv"
    
    print(f"csv_filename: {csv_filename}")
    results_list = []
    
    for i in tqdm(range(start_index, end_index)):
        image_file_name = images_file_names[i]
        # print(image_file_name)
        img_path = os.path.join(dataDir, dataType, image_file_name)
        with Image.open(img_path) as img:
            img_array = np.array(img)
        img_gt = ground_truth_dict[image_file_name]           
        img_resized, gt_resized = resize_image_and_bboxes(img_array,
                                                            img_gt,
                                                            max_shift=max_shift,
                                                            stride=stride,
                                                            cyclic_shift=cyclic_flag,
                                                            target_size=(334, 200))
        
        shifted_imgs, shifted_ground_truth = create_shifted_imgs(img_resized,
                                                                    gt_resized,
                                                                    max_shift=max_shift,
                                                                    stride=stride,
                                                                    cyclic_shift=cyclic_flag)
        # print(len(shifted_imgs))
        
        predictions = []
        for img in shifted_imgs:
            
            try:
                predictions.append(inference_detector(model, img))
            except Exception as e:
                print(f"inference failed on image {image_file_name}")
                print(e)
                continue

        formatted_predictions = format_predictions(predictions)
        formatted_gt = format_ground_truths(shifted_ground_truth)
        try:
            cpe = center_position_error(video_predicts=formatted_predictions,
                                                    video_ground_truths=formatted_gt,
                                                    predict_score_threshold=0.4,
                                                    iou_threshold=0.5)
            sre = scale_and_ratio_error(video_predicts=formatted_predictions,
                                                video_ground_truths=formatted_gt,
                                                predict_score_threshold=0.4,
                                                iou_threshold=0.5)
            
            cpe_results_dict = handle_cpe_error_dict(cpe)
            sre_results_dict = handle_sre_error_dict(sre)
            
            combined_dict = {'Image File Name': image_file_name, **cpe_results_dict, **sre_results_dict}
            results_list.append(combined_dict)
        except Exception as e:
            print(f"couldn't run tests on image {image_file_name}")
            print(e)
            continue
        
        # Write results to CSV every 500 cycles
        if (i - start_index + 1) % 500 == 0:
            if (i + 1) == 500:
                mode = 'w'
            else:
                mode = 'a'
            write_to_csv(csv_filename, results_list, mode=mode)    
            results_list = []

    # Write remaining results to CSV
    if len(results_list) > 0:
        write_to_csv(csv_filename, results_list, mode='a')
        
        
# if __name__ == "__main__":
    
    # args = parse_args()

    # is_afc = args.is_afc
    # max_shift = args.max_shift
    # stride = args.stride
    # random_weights=args.random_weights
    # cyclic_flag = args.is_cyclic
    # start_index = args.start_index
    # is_afc_fpn = args.is_afc_fpn
    # gpu_num = args.gpu_num
    
    # test_runner(is_afc = args.is_afc,
    #             max_shift = args.max_shift,
    #             stride = args.stride,
    #             random_weights=args.random_weights,
    #             cyclic_flag = args.is_cyclic,
    #             start_index = args.start_index,
    #             is_afc_fpn = args.is_afc_fpn,
                # gpu_num = args.gpu_num)
    
    ##for debug
    # is_afc = False
    # max_shift = 1
    # stride = 1
    # random_weights=False
    # cyclic_flag = True
    # start_index = 0
    # is_afc_fpn = False
    # gpu_num = 0
    
    # test_runner(is_afc = False,
    #             max_shift = 1,
    #             stride = 1,
    #             random_weights=False,
    #             cyclic_flag = True,
    #             start_index = 0,
    #             is_afc_fpn = False,
    #             gpu_num = 0)