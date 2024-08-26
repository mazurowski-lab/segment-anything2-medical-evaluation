import sys
sys.path.append('segment-anything-2')

import torch
from sam2.build_sam import build_sam2_video_predictor

from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import LineString, MultiLineString, Polygon, Point, GeometryCollection
from skimage.morphology import medial_axis
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation
from skimage.measure import label
from prompt_funcs import prompt_generating_func

import argparse
import os
import cv2
import json
import random
import matplotlib.pyplot as plt
import numpy as np
# Fix randomness in prompt selection
np.random.seed(1) 
def IOU(pm, gt):
    a = np.sum(np.bitwise_and(pm, gt))
    b = np.sum(pm) + np.sum(gt) - a #+ 1e-8 
    # Per our evaluation critera, slice with empty GT will be ignored
    if np.sum(gt) == 0: return -1
    else:
        return a / b

# Internal loop for running propagation
def _run_prop(start_frame_idx, reverse=False):
    iou_volume
    for frame_idx, object_ids, preds in predictor.propagate_in_video(state, start_frame_idx=start_frame_idx, reverse=reverse):
        curr_mask = cv2.imread(os.path.join(input_dir_mask, '%.5i.png' % (frame_idx)), 0)

        # Convert to H*W*num_cls format
        if num_class > 1:
            mask_one_hot = (np.arange(1, num_class+1) == curr_mask[...,None]).astype(int) 
        else: 
            mask_one_hot = np.array(curr_mask > 0, dtype=int)
        if len(mask_one_hot.shape) < 3:
            mask_one_hot = mask_one_hot[:,:,np.newaxis] # height*depth*1, to consistent with multi-class setting

        preds = np.array(preds.cpu()>0, dtype=int)
        
        mask_cls = mask_one_hot[:,:,cls]
        iou = IOU(preds, mask_cls)
        iou_volume.append(iou)
    return iou_volume

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="SAG segmentor for medical images")
    parser.add_argument("--init-path", default="../publicdata/SAM2/data_3D", type=str, help="the path of the dataset")
    parser.add_argument("--dataset", default="MRI-Heart", type=str, help="the path of the dataset")
    parser.add_argument("--num-class", default=1, type=int, help="number of class for this dataset")
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()
    
    # Set up model
    checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    # Set up dataset
    dataset = args.dataset
    num_class = args.num_class
    input_img_dir = os.path.join(args.init_path, '%s/images' % dataset)
    input_seg_dir = os.path.join(args.init_path, '%s/masks' % dataset)

    print(input_img_dir)
    print(input_seg_dir)

    # Running
    mask_list = os.listdir(input_seg_dir)
    print('# of dataset', len(mask_list))

    MAX_SLICE = 1000
    
    # We evaluate all modes combined
    #for frame_mode in [1,2,3,4]:
    #    for prompt_mode in [1,2,3,5]:
    for frame_mode in [2]:
        for prompt_mode in [1]:
            iou_log = []
            for im_idx, im_name in enumerate(mask_list):
                if 'DS_Store' in im_name:
                    continue
                print('Reading', input_img_dir, im_name, frame_mode, prompt_mode)
                
                # Find prompts based on mode
                input_dir = os.path.join(input_img_dir, im_name)
                input_dir_mask = os.path.join(input_seg_dir, im_name)

                prompts, selected_idx, masks = prompt_generating_func(input_dir_mask, num_class, \
                                               frame_mode=frame_mode, prompt_mode=prompt_mode)

                # Get output based on prompt type
                iou_cls = []
                for cls in range(num_class):
                    iou_volume = []

                    prompt_cls = prompts[cls] # the prompts for this class
                    selected_idx_cls = selected_idx[cls] # the selected frames at the begining 
                    selected_mask_cls = masks[cls]

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = predictor.init_state(input_dir)

                        all_empty = True
                        for pidx in range(len(selected_idx_cls)):
                            prompt = prompt_cls[pidx]
                            if prompt_mode != 5 and len(prompt) == 0:
                                continue
                            else:
                                all_empty = False
                            if prompt_mode in [1,2]:
                                pc = prompt[:,:2]
                                pl = prompt[:, -1]
                                frame_idx, object_ids, preds = predictor.add_new_points_or_box(state, selected_idx_cls[pidx], 0, pc, pl)
                            elif prompt_mode == 3:
                                box = prompt
                                frame_idx, object_ids, preds = predictor.add_new_points_or_box(state, selected_idx_cls[pidx], 0, box=box)
                            elif prompt_mode == 5:
                                frame_idx, object_ids, preds = predictor.add_new_mask(state, selected_idx_cls[pidx], 0, selected_mask_cls[pidx])
                            print('mode %s: prompt %s, slice %s' % (frame_mode, prompt, selected_idx_cls[pidx]))

                        if all_empty:
                            iou_volume = [-1]*MAX_SLICE
                        
                        # If > 0, mask is empty so skip the caseA
                        if len(iou_volume) == 0:
                            if args.bidirectional:
                                # In multi-frame mode, we always select middle slice
                                if frame_mode == 4 and len(selected_idx_cls) > 1:
                                    pidx = 1
                                iou_volume += _run_prop(selected_idx_cls[pidx], reverse=True)
                                iou_volume.reverse()

                                # Re-add prompts
                                predictor.reset_state(state)
                                for pidx in range(len(selected_idx_cls)):
                                    prompt = prompt_cls[pidx]
                                    if prompt_mode in [1,2]:
                                        pc = prompt[:,:2]
                                        pl = prompt[:, -1]
                                        frame_idx, object_ids, preds = predictor.add_new_points_or_box(state, selected_idx_cls[pidx], 0, pc, pl)
                                    elif prompt_mode == 3:
                                        box = prompt
                                        frame_idx, object_ids, preds = predictor.add_new_points_or_box(state, selected_idx_cls[pidx], 0, box=box)
                                    elif prompt_mode == 5:
                                        frame_idx, object_ids, preds = predictor.add_new_mask(state, selected_idx_cls[pidx], 0, selected_mask_cls[pidx])

                                iou_volume += _run_prop(selected_idx_cls[pidx]+1)
                            else:
                                iou_volume += _run_prop(0)
                        
                        # Pad score with zeros
                        if len(iou_volume) < MAX_SLICE:
                            iou_volume += [-1] * (MAX_SLICE - len(iou_volume))

                    iou_cls.append(iou_volume)
                iou_log.append(iou_cls)

            iou_log = np.array(iou_log)
            for cls in range(iou_log.shape[1]):
                tmp = iou_log[:,cls,:]
                print('cls', cls, np.mean(tmp[tmp>=0]))
