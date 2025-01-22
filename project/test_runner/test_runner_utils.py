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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from video_object_detection_metrics.center_position_error import center_position_error
from video_object_detection_metrics.scale_and_retio_error import scale_and_ratio_error
from video_object_detection_metrics.fragment_error import fragment_error
from project_utils.ideal_lpf import subpixel_shift
import time

print("\n".join(sys.path))
print(os.listdir(os.getcwd()))

# sys.path.append(os.path)
# from video_object_detection_metrics import center_position_error
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

def resize_image_and_bboxes(image,
                            annotations,
                            max_shift: int = 0,
                            stride: int = 1,
                            target_size=(1333,800),
                            cyclic_shift=False):
    if cyclic_shift:
        max_shift = 0
        stride = 0
    #resize the image to the size the network take as input, so shift of x pixels will be the same inside the network
    target_size = (int(target_size[0] + max_shift*stride), int(target_size[1] + max_shift*stride))
    # print(f"target size: {target_size}\n")
    # print(type(target_size), type(target_size[0]))
    # Resize the image
    resized_image = cv2.resize(image, target_size)
    
    # Calculate scaling factors for resizing bounding boxes
    scale_x = target_size[0] / image.shape[1]
    scale_y = target_size[1] / image.shape[0]
    
    # Resize and adjust bounding boxes
    resized_annotations = []
    for annotation in annotations:
        bbox = annotation['bbox']
        resized_bbox = [
            int(bbox[0] * scale_x),  # x
            int(bbox[1] * scale_y),  # y
            int(bbox[2] * scale_x),  # width
            int(bbox[3] * scale_y)   # height
        ]
        resized_annotation = {
            'bbox': resized_bbox,
            'category_id': annotation['category_id'],
            'category_name': annotation['category_name']
        }
        resized_annotations.append(resized_annotation)
    
    return resized_image, resized_annotations

def create_shifted_imgs(img, ground_truth, max_shift, stride, cyclic_shift=False, up=4, up_method='ideal'):
    # Determine if max_shift is fractional
    is_fractional = isinstance(max_shift, float) and not max_shift.is_integer()
    max_shift = int(max_shift) if not is_fractional else max_shift  # Convert to int for consistency

    shifted_imgs = []
    shifted_ground_truth = []

    if not is_fractional:
        # Integer shifts
        width, height = img.shape[0], img.shape[1]
        cropped_width = width - max_shift * stride
        cropped_height = height - max_shift * stride

        for i in range(max_shift + 1):
            for j in range(max_shift + 1):
                shift_x = i * stride
                shift_y = j * stride

                if cyclic_shift:
                    # Use torch.roll for cyclic shifts
                    shifted_img = torch.roll(torch.tensor(img), shifts=(-shift_x, -shift_y), dims=(0, 1)).numpy()
                else:
                    if shift_x + cropped_width <= width and shift_y + cropped_height <= height:
                        shifted_img = img[shift_x:shift_x + cropped_width, shift_y:shift_y + cropped_height]
                    else:
                        continue  # Skip if the shift exceeds the image boundaries

                shifted_imgs.append(shifted_img)
                shifted_annotations = []
                for annotation in ground_truth:
                    bbox = annotation['bbox']
                    shifted_bbox = [
                        bbox[0] - shift_x,
                        bbox[1] - shift_y,
                        bbox[2],
                        bbox[3]
                    ]
                    shifted_annotations.append({
                        'bbox': shifted_bbox,
                        'category_id': annotation['category_id'],
                        'category_name': annotation['category_name']
                    })
                shifted_ground_truth.append(shifted_annotations)
    else:
        # Fractional shifts
        img = torch.from_numpy(img).float()
        # assert isinstance(img, torch.Tensor), "Subpixel shifting requires input images as PyTorch tensors."

        # Generate fractional shift values and skip integers
        # fractional_shifts = [i / up for i in range(int(max_shift * up) + 1) if (i / up).is_integer() == False]
        img = img.permute(2, 0, 1) if img.ndim == 3 else img.unsqueeze(0)
        # print(f"before subpixel: {img.shape}")
        fractional_shifts = [0.25, 1.25, 2.25]
        integer_shifts = [0,1,2]

        for i,frac_shift_x in enumerate(fractional_shifts):
            for j,frac_shift_y in enumerate(fractional_shifts):
                try:
                    shifted_img = subpixel_shift(
                        img.unsqueeze(0), up=up, shift_x=int(frac_shift_x * up), shift_y=int(frac_shift_y * up)
                    )[0]
                    
                    shifted_img = shifted_img.squeeze().numpy() if shifted_img.shape[0] == 1 else shifted_img.permute(1, 2, 0).numpy()
                    
                except Exception as e:
                    print(e)
                    print(img.shape)
                    print(int(frac_shift_x * up), int(frac_shift_y * up))
                # print(f"subpixel_shift(img.unsqueeze(0), up={up}, shift_x={int(frac_shift_x * up)}, shift_y={int(frac_shift_y * up)}")
                shifted_imgs.append(shifted_img)
                shifted_annotations = []
            
                for annotation in ground_truth:
                    try:
                        bbox = annotation['bbox']
                        shifted_bbox = [
                            bbox[0] - integer_shifts[i],
                            bbox[1] - integer_shifts[j],
                            bbox[2],
                            bbox[3]
                        ]
                        shifted_annotations.append({
                            'bbox': shifted_bbox,
                            'category_id': annotation['category_id'],
                            'category_name': annotation['category_name']
                        })
                    except Exception as e:
                        print(e)
                        print(img.shape)
                        print(annotation)
                        raise(e)
                shifted_ground_truth.append(shifted_annotations)
    return shifted_imgs, shifted_ground_truth


# def create_shifted_imgs(img, ground_truth, max_shift, stride, cyclic_shift=False):
#     # img format needs to be (width, height, 3), values in range [0,1]
#     # max_shift is the number of output images
#     # print(f"img shape: {img.shape}")
#     width, height = img.shape[0], img.shape[1]
#     cropped_width = width - max_shift*stride
#     cropped_height = height - max_shift*stride
#     # print(f"width={width}, height = {height}")
#     shifted_imgs = []
#     shifted_ground_truth = []
#     for i in range(max_shift + 1):
#         for j in range(max_shift + 1):
#             shift_x = i * stride
#             shift_y = j * stride
#             if cyclic_shift:
#                 shifted_img = np.zeros_like(img)
#                 if len(img.shape) == 3:  # Color image (width, height, 3)
#                     shifted_img[0:width - shift_x, 0:height - shift_y, :] = img[shift_x:, shift_y:, :]
#                     shifted_img[width - shift_x:, 0:height - shift_y, :] = img[0:shift_x, shift_y:, :]
#                     shifted_img[0:width - shift_x, height - shift_y:, :] = img[shift_x:, 0:shift_y, :]
#                     shifted_img[width - shift_x:, height - shift_y:, :] = img[0:shift_x, 0:shift_y, :]
#                 else:  # Black and white image (width, height)
#                     shifted_img[0:width - shift_x, 0:height - shift_y] = img[shift_x:, shift_y:]
#                     shifted_img[width - shift_x:, 0:height - shift_y] = img[0:shift_x, shift_y:]
#                     shifted_img[0:width - shift_x, height - shift_y:] = img[shift_x:, 0:shift_y]
#                     shifted_img[width - shift_x:, height - shift_y:] = img[0:shift_x, 0:shift_y] 
#             else:
#                 if shift_x + cropped_width <= width and shift_y + cropped_height <= height:
#                     if len(img.shape) == 3:  # Color image (width, height, 3)
#                         shifted_img = img[shift_x:shift_x + cropped_width, shift_y:shift_y + cropped_height, :]
#                     else:  # Black and white image (width, height)
#                         shifted_img = img[shift_x:shift_x + cropped_width, shift_y:shift_y + cropped_height]
#                 else:
#                     continue  # Skip if the shift exceeds the image boundaries
#             shifted_imgs.append(shifted_img)
#             # print(f"shifted img shape = {shifted_img.shape}")
#             # Adjust ground truth annotations
#             shifted_annotations = []
#             for annotation in ground_truth:
#                 bbox = annotation['bbox']
#                 if cyclic_shift:
#                     diff_x = 0
#                     diff_y = 0
#                     if bbox[0] - shift_x < 0:
#                         diff_x = shift_x - bbox[0]
#                     if bbox[1] - shift_y < 0:
#                         diff_y = shift_y - bbox[1]
#                     new_top_x = max(0, bbox[0] - shift_x)
#                     new_top_y = max(0, bbox[1] - shift_y)
#                     shifted_bbox = [
#                         new_top_x,
#                         new_top_y,
#                         bbox[2] - diff_x,
#                         bbox[3] - diff_y
#                     ]
#                 else:
#                     shifted_bbox = [
#                         bbox[0] - shift_x,
#                         bbox[1] - shift_y,
#                         bbox[2],
#                         bbox[3]
#                     ]
#                 shifted_annotation = {
#                     'bbox': shifted_bbox,
#                     'category_id': annotation['category_id'],
#                     'category_name': annotation['category_name']
#                 }
#                 shifted_annotations.append(shifted_annotation)
#             shifted_ground_truth.append(shifted_annotations)

#     return shifted_imgs, shifted_ground_truth

def get_images(path_to_imgs): #TODO: move to utils
    images_tensor = []

    # Iterate through the JPEG files in the folder
    for filename in tqdm(os.listdir(path_to_imgs)):
        if filename.endswith(".jpg"):
            filepath = os.path.join(path_to_imgs, filename)

            # Open the image using PIL
            with Image.open(filepath) as img:
                if img.mode != "RGB":
                    continue
                # Convert to a PyTorch tensor with desired dimensions
                # print(img)
                # img = img.resize((W, H), Image.LANCZOS)  # Use desired interpolation method
                img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1))  # Rearrange for HxWxC format
                # img_tensor = img_tensor.to(device)
                # Add the image tensor to the existing tensor
                # images_tensor = torch.cat((images_tensor, img_tensor.unsqueeze(0)), dim=0)
                images_tensor.append(img_tensor)
    return images_tensor

def draw_bounding_boxes(image, bboxes, output_path):
    """
    Draws bounding boxes on an image and saves the result to a file.

    Args:
    - image (PIL.Image.Image): The input image.
    - bboxes (list): List of bounding boxes in the format [id, x1, y1, x2, y2].
    - output_path (str): Path to save the output image.

    Returns:
    None
    """
    
    image = Image.fromarray(image)
    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Draw each bounding box
    for bbox in bboxes:
        _, x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Save the image with bounding boxes
    image.save(output_path)

# Convert predictions to the required format
def format_predictions(predictions, score_threshold=0):
    formatted_predictions = []
    for frame_predictions in predictions:
        frame_result = []

        for c in frame_predictions[0]:
            for bbox in c:
                # x, y, w, h, score = bbox
                x1, y1, x2, y2, score = bbox
                if score >= score_threshold:
                    # x1 = x
                    # y1 = y
                    # x2 = x + w
                    # y2 = y + h
                    frame_result.append([score, x1, y1, x2, y2])
        formatted_predictions.append(frame_result)
    return formatted_predictions

# Convert ground truths to the required format
def format_ground_truths(ground_truths):
    formatted_ground_truths = []
    for frame_ground_truth in ground_truths:
        person_id = 1  # Assign a unique person_id to each ground truth annotation
        frame_result = []
        for gt in frame_ground_truth:
            x1, y1, w, h = gt['bbox']
            x2 = x1 + w
            y2 = y1 + h
            frame_result.append([person_id, x1, y1, x2, y2])
            person_id += 1
        formatted_ground_truths.append(frame_result)
    return formatted_ground_truths

def handle_cpe_error_dict(data):
    
    std_x_vals = []
    std_y_vals = []
    mean_x_vals = []
    mean_y_vals = []
    mean_dist_vals = []
    worst_case_dist_vals = []
    nan_counter = 0
    
    for values in data.values():
        if isinstance(values, float) and np.isnan(values):
            nan_counter += 1
            continue
        
        std_x, std_y, mean_x, mean_y, mean_dist, worst_case_dist = values
        
        std_x_vals.append(std_x)
        std_y_vals.append(std_y)
        mean_x_vals.append(mean_x)
        mean_y_vals.append(mean_y)
        mean_dist_vals.append(mean_dist)
        worst_case_dist_vals.append(worst_case_dist)

    if len(worst_case_dist_vals) < 1:
        worst_case_dist_vals.append(None)
        
    return {'cpe_std_x': np.mean(std_x_vals),
            'cpe_std_y': np.mean(std_y_vals),
            'CPE': np.mean(np.array(std_x_vals) + np.array(std_y_vals)),
            'cpe_mean_x': np.mean(mean_x_vals),
            'cpe_mean_y': np.mean(mean_y_vals),
            'cpe_dist': np.mean(mean_dist_vals),
            'worst_case_cpe_dist': np.max(worst_case_dist_vals),
            'cpe_nan_counter': nan_counter,
    }


def handle_sre_error_dict(data):
    
    std_scale_vals = []
    std_ratio_vals = []
    mean_scale_vals = []
    mean_ratio_vals = []
    nan_counter = 0
    
    for values in data.values():
        if isinstance(values, float) and np.isnan(values):
            nan_counter += 1
            continue
        
        std_scale, std_ratio, mean_scale, mean_ratio = values
        
        std_scale_vals.append(std_scale)
        std_ratio_vals.append(std_ratio)
        mean_scale_vals.append(mean_scale)
        mean_ratio_vals.append(mean_ratio)
    
    return {'sre_std_x': np.mean(std_scale_vals),
            'sre_std_y': np.mean(std_ratio_vals),
            'SRE': np.mean(np.array(std_scale_vals) + np.array(std_ratio_vals)),
            'sre_mean_x': np.mean(mean_scale_vals),
            'sre_mean_y': np.mean(mean_ratio_vals),
            'sre_nan_counter': nan_counter,
    }





def write_to_csv(filename, data, mode='w'):
    """
    Writes a list of dictionaries to a CSV file.

    Args:
    - filename (str): The name of the file to write to.
    - data (list of dict): The data to write, where each dictionary has the same keys.
    - mode (str): The file mode, 'w' for write (default) or 'a' for append.

    Returns:
    None
    """
    if not data:
        raise ValueError("Data is empty")
    
    # Extract headers from the first dictionary
    headers = data[0].keys()
    
    with open(filename, mode, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write the header only if the mode is 'w' (write) and the file is being created
        if mode == 'w':
            writer.writeheader()
        
        # Write the data rows
        writer.writerows(data)

def extract_params(filename):
    # Extract the max_shift and stride from the filename
    # Assuming filename format is something like "max_shift_X_stride_Y.csv"
    parts = filename.replace('.csv', '').split('_')
    max_shift = int(parts[2])
    stride = int(parts[4])
    return max_shift, stride

def plot_graphs(csv_dir_path):
    # Initialize data storage
    data = defaultdict(list)

    # Folder containing the CSV files
    folder_path = csv_dir_path

    # Read each CSV file and store the necessary data
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            max_shift, stride = extract_params(filename)
            df = pd.read_csv(os.path.join(folder_path, filename))
            data[(max_shift, stride)].append(df)

    # Prepare data for plotting
    average_cpe_results = []
    average_sre_results = []
    sre_results = []
    cpe_results = []

    for key, dfs in data.items():
        combined_df = pd.concat(dfs)
        avg_cpe = combined_df['CPE Results'].mean()
        avg_sre = combined_df['SRE Results'].mean()
        average_cpe_results.append({'max_shift': key[0], 'stride': key[1], 'average_cpe': avg_cpe})
        average_sre_results.append({'max_shift': key[0], 'stride': key[1], 'average_sre': avg_sre})
        sre_results.extend(combined_df['SRE Results'].tolist())
        cpe_results.extend(combined_df['CPE Results'].tolist())

    average_cpe_df = pd.DataFrame(average_cpe_results)
    average_sre_df = pd.DataFrame(average_sre_results)

    # Create a directory to save the images
    output_dir = os.path.join(folder_path, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Bar graph for average CPE Results
    fig_cpe = px.bar(average_cpe_df, x=['max_shift', 'stride'], y='average_cpe', 
                     title='Average CPE Results by Max Shift and Stride')
    fig_cpe.write_image(os.path.join(output_dir, "average_cpe_results.png"))

    # 2. Bar graph for average SRE Results
    fig_sre = px.bar(average_sre_df, x=['max_shift', 'stride'], y='average_sre', 
                     title='Average SRE Results by Max Shift and Stride')
    fig_sre.write_image(os.path.join(output_dir, "average_sre_results.png"))

    # 3. Histogram for SRE Results
    hist_sre = px.histogram(average_sre_df, x='average_sre', color=['max_shift', 'stride'], 
                            title='SRE Results Histogram', histnorm='percent')
    hist_sre.write_image(os.path.join(output_dir, "sre_results_histogram.png"))

    # 4. Histogram for CPE Results
    hist_cpe = px.histogram(average_cpe_df, x='average_cpe', color=['max_shift', 'stride'], 
                            title='CPE Results Histogram', histnorm='percent')
    hist_cpe.write_image(os.path.join(output_dir, "cpe_results_histogram.png"))



def fpn_check():
    
    import os
    print(os.environ['PATH'])

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

    from our_demo import resize_image_and_bboxes, create_shifted_imgs, format_predictions, format_ground_truths, handle_cpe_error_dict, handle_sre_error_dict, write_to_csv


    from collections import defaultdict
    import os
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    from mmdet.models.backbones.convnext_afc import ConvNeXtAFC
    from mmdet.models.necks.fpn import AFC_FPN, FPN
    from mmcv.cnn import ConvModule, AFC_ConvModule

    img_path = "/media/ssd/Datasets/coco/val2017/000000581781.jpg"

    backbone = ConvNeXtAFC(in_chans=3,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            out_indices=[0, 1, 2, 3],
            activation='up_poly_per_channel',
            activation_kwargs=dict(in_scale=7, out_scale=7, train_scale=True),
            blurpool_kwargs=dict(filter_type='ideal', scale_l2=False),
            normalization_type='CHW2',
            stem_activation_kwargs=dict(
                in_scale=7, out_scale=7, train_scale=True, cutoff=0.75),
            normalization_kwargs=dict(),
            stem_mode='activation_residual',
            stem_activation='lpf_poly_per_channel')

    afc_fpn = AFC_FPN(in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=5)

    device = 'cuda:1'

    backbone.to(device=device).eval()
    afc_fpn.to(device=device).eval()

    print(afc_fpn)

    baseline_fpn = FPN(in_channels=[96, 192, 384, 768],
                    out_channels=256,
                    num_outs=5)

    baseline_fpn.to(device=device).eval()

    from torchvision import transforms

    # load example image
    interpolation = transforms.InterpolationMode.BICUBIC
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=interpolation),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(device)
    #image = transform(image).to(device=device)

    feat_maps = backbone.forward(image)

    fpn_output = afc_fpn.forward(feat_maps)

    from project_utils.ideal_lpf import UpsampleRFFT
    @torch.no_grad()
    def shift_and_compare(backbone, fpn, image, shift_x, shift_y):
        """
        Cyclic-Shifts the image, extracts features, upsamples, shifts back, and compares.

        Args:
        model: The PyTorch model to use for feature extraction.
        image: The input image tensor.
        shift_x: Horizontal shift amount.
        shift_y: Vertical shift amount.

        Returns:
        A tuple containing:
            - The original feature map.
            - The shifted and reversed feature map.
            - The difference between the two feature maps.
        """

        # Shift the image cyclically
        shifted_image = torch.roll(image, shifts=(shift_x, shift_y), dims=(3, 2))

        # Get feature maps from the model
        if fpn is None:
            feature_map = list(backbone.forward(image))
            shifted_feature_map = list(backbone.forward(shifted_image))
        else:
            feature_map = list(fpn.forward(backbone.forward(image)))
        
        # for i in range(len(feature_map)):
        #   print(feature_map[i].shape)
        
            shifted_feature_map = list(fpn.forward(backbone.forward(shifted_image)))

        # Upsample to the original image size
        for i in range(len(feature_map)):
            size_ratio = int(image.shape[-1] / feature_map[i].shape[-1])
            feature_map[i] = UpsampleRFFT(up=size_ratio)(feature_map[i])
            # print(size_ratio)
            shifted_feature_map[i] = UpsampleRFFT(up=size_ratio)(shifted_feature_map[i])

            # Reverse the shift
            shifted_feature_map[i] = torch.roll(shifted_feature_map[i], shifts=(-shift_x, -shift_y), dims=(3, 2))

            # Featuremap shift-equivariance diff
            difference = torch.abs(shifted_feature_map[i] - feature_map[i])
            print("featuremap avg diff: ", torch.mean(difference))

            # Feature-vector invariance / sum-shift invariance
            feature_vec = torch.mean(feature_map[i], dim=(2, 3))
            shifted_feature_vec = torch.mean(shifted_feature_map[i], dim=(2, 3))

            print("feature vector diff: ", torch.mean(torch.abs(feature_vec - shifted_feature_vec)))

    shift_and_compare(backbone=backbone,
                fpn=afc_fpn,
                # fpn=baseline_fpn,
                image=image,
                shift_x=3,
                shift_y=3)


if __name__ == "__main__":
    fpn_check()

"""
if __name__ == "__main__":
    # Choose to use a config and initialize the detector
    config = "/home/nirt/folder/Swin-Transformer-Object-Detection/mask_rcnn_convnext_afc_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py"
    # Setup a checkpoint file to load
    checkpoint = '/media/ssd/hagaymi/convnext_afc_maskrcnn/work_dirs/mask_rcnn_convnext_afc_tiny_ideal_up_poly_per_channel_scale_7_7_train_chw2_stem_mode_activation_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k/latest.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')
    
    PATH_TO_COCO_VAL_IMAGES = '/media/ssd/Datasets/coco/val2017'
    img = os.path.join(PATH_TO_COCO_VAL_IMAGES, '000000580418.jpg')
    result = inference_detector(model, img)

    bboxes = result[0]
    out_path = "123.png"
    # print(bboxes)
    # print(result)

    # visualize_and_save_bounding_boxes(bboxes[20], img, out_path)
    #show_result_pyplot(model, img, result, score_thr=0.3, out_file="123.png")
    # show_result_pyplot(model=model,
    #                    img=os.path.join(PATH_TO_COCO_VAL_IMAGES, '000000580418.jpg'),
    #                    result=result,
    #                    score_thr=0.3)

    # output_path = '222.png'
    # plt.savefig(output_path)
    # plt.show()
    
    # model.show_result(
    #     img,
    #     result,
    #     score_thr=0.7,
    #     show=True,
    #     wait_time=0,
    #     win_name='title',
    #     bbox_color=(72, 101, 241),
    #     text_color=(200, 200, 200),
    #     out_file='123test.png')
    

    # Define paths
    dataDir = '/media/ssd/Datasets/coco'  # Adjust this to your actual dataset path
    dataType = 'val2017'  # Only process validation data
    annFile = os.path.join(dataDir, 'annotations', f'instances_{dataType}.json')
    imageDir = os.path.join(dataDir, dataType)

    # Define a directory to save the ground truth data
    output_dir = os.path.join(dataDir, 'ground_truths', dataType)
    # Dictionary to store all ground truths with file names as keys
    ground_truth_dict = {}
    # Initialize COCO API
    coco = COCO(annFile)

    # Get all image IDs
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
    
    
    test_image = '000000580418.jpg'
    # test_image = "000000024021.jpg"
    path_to_test_image = os.path.join(PATH_TO_COCO_VAL_IMAGES, test_image)
    test_image_gt = ground_truth_dict[test_image]
    with Image.open(path_to_test_image) as img:
        # print(img.shape)
        img_array = np.array(img)
        # print(img_array)
        # print(img_array.shape)
        
    test_img_resized, test_gt_resized = resize_image_and_bboxes(img_array,
                                                                test_image_gt,
                                                                max_shift=2,
                                                                stride=5)
    # print(test_img_resized.shape)
    
    test_shifted_imgs, test_shifted_ground_truth = create_shifted_imgs(test_img_resized,
                                                                       test_gt_resized,
                                                                       max_shift=1,
                                                                       stride=1)
    # test_predictions = inference_detector(model, test_shifted_imgs)
    print(type(test_img_resized))
    print(type(test_img_resized[0][0][0]))
    # test_img_resized = torch.tensor(test_img_resized)
    print(type(test_img_resized))
    print(type(test_img_resized[0][0][0]))
    test_predictions = inference_detector(model, test_img_resized)
    # test_predictions = inference_detector(model, test_shifted_imgs)
    # print(test_predictions[0][0])
    # print(f"test_shifted_ground_truth = {test_shifted_ground_truth}")
    

    formatted_predictions = format_predictions(test_predictions)
    formatted_gt = format_ground_truths(test_shifted_ground_truth)
    print(formatted_gt)
    draw_bounding_boxes(test_shifted_imgs[0],formatted_gt[0], "gt_test.jpg")
    print(f"\n\npred[0] = {formatted_predictions[0]}\n\n")
    print(f"\n\npred[1] = {formatted_predictions[1]}\n\n")
    print(f"\n\ngt = {formatted_gt[0][7:9]}")
    print(f"{test_gt_resized[7:9]}\n\n")
    
    
    error = center_position_error(video_predicts=formatted_predictions,
                                  video_ground_truths=formatted_gt,
                                  predict_score_threshold=0,
                                  iou_threshold=0.2)
    ratio_error = scale_and_ratio_error(video_predicts=formatted_predictions,
                                        video_ground_truths=formatted_gt,
                                        predict_score_threshold=0,
                                        iou_threshold=0.2)
    
    frag_error = fragment_error(video_predicts=formatted_predictions,
                                        video_ground_truths=formatted_gt,
                                        predict_score_threshold=0,
                                        iou_threshold=0.2)
    
    error1 = np.mean(np.array(list(error.values())))

    error2, number_of_false = calc_mean_of_error_dict(error)

    print(f"\n\nerror1 with max shift 2 stride 5 = {error1}\n\n")
    print(f"\n\nerror2 with max shift 2 stride 5 = {error2}\n\n")
    print(f"\n\nfalse values with max shift 2 stride 5 = {number_of_false}\n\n")
    print(f"\n\nratio error with max shift 2 stride 5 = {ratio_error}\n\n")
    print(f"\n\nfrag_error with max shift 2 stride 5 = {frag_error}\n\n")


    # print(test_image_gt)
    
    test_img_resized, test_gt_resized = resize_image_and_bboxes(img_array,
                                                                test_image_gt,
                                                                max_shift=2,
                                                                stride=1)
    # print(test_img_resized.shape)

    test_shifted_imgs, test_shifted_ground_truth = create_shifted_imgs(test_img_resized,
                                                                       test_gt_resized,
                                                                       max_shift=2,
                                                                       stride=1)
    test_predictions = inference_detector(model, test_shifted_imgs)
    # print(test_predictions[0][0])
    # print(f"test_shifted_ground_truth = {test_shifted_ground_truth}")
    formatted_predictions = format_predictions(test_predictions)
    formatted_gt = format_ground_truths(test_shifted_ground_truth)
    # print(f"pred = {formatted_predictions[0]}")
    # print(f"gt = {formatted_gt[0]}")
    error = center_position_error(video_predicts=formatted_predictions,
                                  video_ground_truths=formatted_gt,
                                  predict_score_threshold=0,
                                  iou_threshold=0.2)
    
    ratio_error = scale_and_ratio_error(video_predicts=formatted_predictions,
                                        video_ground_truths=formatted_gt,
                                        predict_score_threshold=0,
                                        iou_threshold=0.2)
    
    frag_error = fragment_error(video_predicts=formatted_predictions,
                                        video_ground_truths=formatted_gt,
                                        predict_score_threshold=0,
                                        iou_threshold=0.2)
    error = np.mean(np.array(list(error.values())))

    print(f"\n\nerror with max shift 2 stride 1 = {error}\n\n")
    print(f"\n\nratio_error with max shift 2 stride 1 = {ratio_error}\n\n")
    print(f"\n\nfrag_error with max shift 2 stride 1 = {frag_error}\n\n")
    
"""
