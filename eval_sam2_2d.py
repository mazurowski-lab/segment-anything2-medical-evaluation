import sys
sys.path.append('segment-anything-2')

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import LineString, MultiLineString, Polygon, Point, GeometryCollection
from skimage.morphology import medial_axis
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation
from skimage.measure import label
from prompt_funcs import gen_prompt_for_single_slice_func

import argparse
import os
import cv2
import json
import random
import matplotlib.pyplot as plt
import numpy as np
# Fix randomness in prompt selection
np.random.seed(1) 
#This is a helper function that should not be called directly

def IOU(pm, gt):
    a = np.sum(np.bitwise_and(pm, gt))
    b = np.sum(pm) + np.sum(gt) - a #+ 1e-8 
    if b == 0:
        return -1
    else:
        return a / b

def IOUMulti(y_pred, y):
    score = 0
    numLabels = np.max(y)
    if np.max(y) == 1:
        score = IOU(y_pred, y)
        return score
    else:
        count = 1
        for index in range(1,numLabels+1):
            curr_score = IOU(y_pred[y==index], y[y==index])
            print(index, curr_score)
            if curr_score != -1:
                score += curr_score
                count += 1
        return score / (count - 1) # taking average


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="SAG segmentor for medical images")
    parser.add_argument("--init-path", default="../publicdata/SAM2", type=str, help="the path of the dataset")
    parser.add_argument("--dataset", default="MRI-Heart", type=str, help="the specific dataset")
    parser.add_argument("--num-class", default=1, type=int, help="the number of classes for this dataset")
    args = parser.parse_args()
    
    # Set up model
    checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # Set up dataset
    dataset = args.dataset
    num_class = args.num_class
    input_img_dir = os.path.join(args.init_path, '%s/images' % dataset)
    input_seg_dir = os.path.join(args.init_path, '%s/masks' % dataset)

    print(input_img_dir)
    print(input_seg_dir)

    # Running
    dc_log, names = [], []
    mask_list = os.listdir(input_seg_dir)
    print('# of dataset', len(mask_list))
    
    # VIS: now VIS function is separted into another file. Only provide mask if neede
    vis = False
    # Change to [name1, name2, ...] if only need to run on a few samples
    im_list = None

    for im_idx, im_name in enumerate(mask_list):
        preds_mask_full = []
        # Skip non-selected images if specified
        if im_list is not None:
            if im_name not in im_list:
                continue
        print('Predicting', im_name)

        if 'DS_Store' in im_name or 'segmentations2D' in im_name:
            continue

        # Read image and mask
        try:
            input_mask = cv2.imread(os.path.join(input_seg_dir, im_name), 0)  
        except:
            print('Cannot read mask', im_name)
            continue
    
        # Skip images with empty mask
        if np.max(input_mask) == 0:
            print('Empty mask')
            print('*****')
            continue
        
        
        try:
            input_image = Image.open(os.path.join(input_img_dir, im_name)).convert("RGB")
        except:
            input_image = Image.open(os.path.join(input_img_dir, im_name.replace('png', 'jpg'))).convert("RGB")

        input_array = np.array(input_image)
        input_array = np.uint8(input_array / np.max(input_array) * 255)
        print('Number of labels', np.max(input_mask))
        print('Image maximum', np.max(input_array))
        
        # Mask has to be float
        dc_class_tmp = []
        for cls in range(num_class):
            dc_prompt_tmp = []

            # Start prediction for each class
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                input_image_pil = Image.open(os.path.join(input_img_dir, im_name.replace('png', 'jpg'))).convert("RGB")
                predictor.set_image(input_image_pil)

                # 4 modes for now
                for mode in range(1,5):
                    prompt, mask_cls = gen_prompt_for_single_slice_func(input_mask, cls, num_class, prompt_mode=mode)

                    
                    # No prompt can be found if mask is empty
                    if prompt is None:
                        #print('Skip b/c mask empty for cls', cls)
                        if num_class == 1:
                            dc_prompt_tmp = [np.nan]
                        else:
                            dc_prompt_tmp = [np.nan] * 4
                        break 
                    
                    # Get output based on prompt type
                    prompt = np.array(prompt)

                    print('mode %s: prompt: %s' % (mode, prompt))
                    if prompt.shape[-1] == 3:
                        pc = prompt[:,:2]
                        pl = prompt[:, -1]
                        preds, _, _ = predictor.predict(point_coords=pc, point_labels=pl)
                    elif prompt.shape[-1] == 4:
                        if len(prompt.shape) == 1:
                            preds, _, _ = predictor.predict(box=prompt)
                        else:
                            preds = None
                            for box in prompt:
                                preds_single, _, _ = predictor.predict(box=box)
                                if preds is None:
                                    preds = preds_single
                                else:
                                    preds += preds_single

                    preds = preds.transpose((1,2,0))
                    # In this paper, we only evaluate SAM with the first channel's output
                    preds_mask_single = np.array(preds[:,:,0]>0,dtype=int)
                    
                    print(preds_mask_single.shape, mask_cls.shape)
                    dc = IOU(preds_mask_single, mask_cls)
                    dc_prompt_tmp.append(dc)
                    print('IoU:', dc)
                    
                    # Track prediction, only used when vis
                    if vis:
                        preds_mask_full.append(np.expand_dims(preds, 0))
                
                # assgin final mask for this class to it
                dc_class_tmp.append(dc_prompt_tmp)
                print('****')
        
        dc_log.append(dc_class_tmp)
        names.append(im_name)
        print('****')
        
        # VIS mode only saves mask and prompt information
        if vis:
            # Final shape: N*H*W*3
            # N = number of predictions. 1 if box prompt, otherwise number of prompts
            # H,W = size of mask
            # 3 = number of outputs per prediction. SAM returns 3 outpus per prompt. 
            #     If no oracle mode, select 0
            #     If oracle mode, select maximum slice. 
            #     You can do that later, or use variable "max_slice"
            if len(preds_mask_full) > 0:
                preds_mask_full = np.concatenate(preds_mask_full)

                # If box:    N*4, N=number of boxes, 4=box coordinate in XYXY format
                # If prompts:N*3, N=number of prmts, 3=cX, cY, pos/neg
                np.save('tmp%s_pred.npy' % im_name[:-4], preds_mask_full)

    if not vis:
        # BRATS labelled class as 1,2,4
        dc_log = np.array(dc_log)
        print(dc_log.shape)
        print(np.nanmean(dc_log, axis=0))
        print(np.nanmean(dc_log))

