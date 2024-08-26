from skimage.measure import label
#Scientific computing 
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import one_hot
import cv2
import torch
import random
#Pytorch packages

def First_frame_finding_func(mask_list,mask_dir,num_class=1,mode=0,frame_num=3):
    '''
    Select the first frame to annotate based on 
    1. frame with edge mask for each class (mask at edge slice)
    2. frame of the center slice of each class's object
    3. frame of largest slice of each class's object
    4. three frame uniformly distributed for each class
    '''
    select_mask_idx = [[] for _ in range(num_class)]
    non_zero_slices = [[] for _ in range(num_class)]
    max_mask_count = [0]*num_class
    for idx, msk in enumerate(mask_list):
        curr_mask = cv2.imread(os.path.join(mask_dir, msk), 0)
        # Convert to H*W*num_cls format
        if num_class > 1:
            mask_one_hot = (np.arange(1, num_class+1) == curr_mask[...,None]).astype(int) 
        else: 
            mask_one_hot = np.array(curr_mask > 0, dtype=int)
        if len(mask_one_hot.shape) < 3:
            mask_one_hot = mask_one_hot[:,:,np.newaxis] # height*depth*1, to consistent with multi-class setting
            
        if mode == 3: # Find slice with largest mask
            for cls in range(num_class):
                mask_cls = np.uint8(mask_one_hot[:,:,cls])
                count = np.sum(mask_cls)
                if count > max_mask_count[cls]:
                    max_mask_count[cls] = count
                    select_mask_idx[cls] = [idx]
        elif mode in [1, 2, 4]:
            for cls in range(num_class):
                mask_cls = np.uint8(mask_one_hot[:, :, cls])
                if np.sum(mask_cls) > 0:
                    non_zero_slices[cls].append(idx)

    if mode == 1 and non_zero_slices:  # Select a edge slice
       for cls in range(num_class):
           if non_zero_slices[cls]:
                first_slice_idx = non_zero_slices[cls][0]
                select_mask_idx[cls] = [first_slice_idx]
    elif mode == 2 and non_zero_slices: # select a center slice
       for cls in range(num_class):
           if non_zero_slices[cls]:
                center_slice_idx = non_zero_slices[cls][len(non_zero_slices[cls]) // 2]
                select_mask_idx[cls] = [center_slice_idx]
    elif mode == 4 and non_zero_slices:  # Uniformly select frame_num non-empty frames for each class
        for cls in range(num_class):
            if  non_zero_slices[cls]:
                step = max(1, len(non_zero_slices[cls]) // (frame_num+1))
                select_mask_idx[cls] = non_zero_slices[cls][::step]
                if len(select_mask_idx[cls]) > frame_num+1:
                    select_mask_idx[cls] = select_mask_idx[cls][1:frame_num+1]
                else:
                    select_mask_idx[cls] = select_mask_idx[cls][:frame_num]
            else:
                select_mask_idx[cls] = []
    #print('Selected slice for prompt is:',select_mask_idx)
    return select_mask_idx


def gen_prompt_for_single_slice_func(selected_mask,cls_idx, num_class=1,prompt_mode=0):
    if num_class > 1:
        mask_one_hot = (np.arange(1, num_class+1) == selected_mask[...,None]).astype(int) 
    else: 
        mask_one_hot = np.array(selected_mask > 0, dtype=int)
    if len(mask_one_hot.shape) < 3:
        mask_one_hot = mask_one_hot[:,:,np.newaxis] # height*depth*1, to consistent with multi-class setting
    mask_cls = mask_one_hot[:,:,cls_idx]

    if np.sum(mask_cls) == 0:
        return None, mask_cls
    
    if prompt_mode == 1:
        # prompt mode 1: find largest region and assign one point prompt
        prompt,_ = get_first_prompt(mask_cls,region_type = 'largest_1',prompt_num=1,max_prompt_num=1)
    elif prompt_mode == 2:
        # prompt mode 2: find all regions and assign one point for each region
        prompt,_ = get_first_prompt(mask_cls,region_type = 'largest_3',prompt_num=3,max_prompt_num=3)
    elif prompt_mode == 3: 
        # prompt mode 3: find largest regions and assign one box prompt
        prompt,_ = get_top_boxes(mask_cls,region_type = 'largest_1',prompt_num=1)
    elif prompt_mode == 4:
        # prompt mode 4: find all regions and assign one box for one region
        prompt,_ = get_top_boxes(mask_cls,region_type =  'largest_3',prompt_num=3)
    elif prompt_mode == 5: 
        # prompt mode 5: directly apply groundth truth mask at first slice
        prompt = -1
    return prompt, mask_cls


def prompt_generating_func(mask_dir, num_class=1, frame_mode=1, prompt_mode=1):
    mask_list = os.listdir(mask_dir)
    mask_list = [i for i in mask_list if not i.startswith('.')]
    mask_list.sort(key=lambda p: int(os.path.splitext(p)[0]))
    select_mask_idx = First_frame_finding_func(mask_list,mask_dir,num_class,frame_mode)
    prompts = []
    masks = []
    for cls_idx, select_idxs in enumerate(select_mask_idx):
        prompts_cls = []
        masks_cls = []
        for select_idx in select_idxs:
            #print(f'current cls to segment: {cls_idx}, selected slice for this class: {select_idx}')
            selected_mask = cv2.imread(os.path.join(mask_dir, mask_list[select_idx]), 0)
            prompt, mask_cls = gen_prompt_for_single_slice_func(selected_mask, cls_idx, num_class, prompt_mode=prompt_mode)
            prompts_cls.append(prompt) # one list for each class  
            masks_cls.append(mask_cls)
        prompts.append(prompts_cls)
        masks.append(masks_cls)
    # prompts format: num_cls * num_frame_selected * 3/4
    # masks format: num_cls * num_frame_selected * mask
    return prompts, select_mask_idx, masks



def random_sum_to(n, num_terms = None):
    '''
    generate num_tersm with sum as n
    '''
    num_terms = (num_terms or r.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[i+1] - a[i] for i in range(len(a) - 1)]



def get_first_prompt(mask_cls,dist_thre_ratio=0,prompt_num=10,max_prompt_num=15,region_type='random'):
    '''
    if region_type = random, we random select one region and generate prompt
    if region_type = all, we generate prompt at each object region
    if region_type = largest_k, we generate prompt at largest k region, k <10
    '''
    if prompt_num==-1:
        prompt_num = random.randint(1, max_prompt_num)
    # Find all disconnected regions
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    #print('num of regions found', region_ids)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids+1):
        #find coordinates of points in the region
        binary_msk = np.where(label_msk==region_id, 1, 0)

        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(mask_cls)
        #print('curr mask over all mask ratio', r)
        ratio_list.append(r)
        regionid_list.append(region_id)
    if len(ratio_list)>0:
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]
    
        if region_type == 'random':
            prompt_num = 1
            regionid_list = [random.choice(regionid_list)] # random choose 1 region
            prompt_num_each_region = [1]
        elif region_type[:7] == 'largest':
            region_max_num = int(region_type.split('_')[-1])
            #print(region_max_num,prompt_num,len(regionid_list))
            valid_region = min(region_max_num,len(regionid_list))
            if valid_region<prompt_num:
                prompt_num = valid_region
                max_prompt_num = min(valid_region,3)
                prompt_num_each_region = valid_region*[1]
                #prompt_num_each_region = random_sum_to(prompt_num,valid_region)
            else:
                prompt_num_each_region = prompt_num*[1]
            regionid_list = regionid_list[:min(valid_region,prompt_num)]
        else:
            prompt_num_each_region = len(regionid_list)*[1]
            max_prompt_num = min(len(regionid_list),3)


        prompt = []
        mask_curr = np.zeros_like(label_msk)
        

        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk==regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk,mask_curr)


            padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
            dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]

            # sort the distances 
            dist_array=sorted(dist_img.copy().flatten())[::-1]
            dist_array = np.array(dist_array)
            # find the threshold:
            dis_thre = max(dist_array[int(dist_thre_ratio*np.sum(dist_array>0))],1)
            #print(np.max(dist_array))
            #print(dis_thre)
            cY, cX = np.where(dist_img>=dis_thre)
            while prompt_num_each_region[reg_id]>0:
                # random select one prompt
                random_idx = np.random.randint(0, len(cX))
                cx, cy = int(cX[random_idx]), int(cY[random_idx])
                prompt.append((cx,cy,1))
                prompt_num_each_region[reg_id] -=1

        #while len(prompt)<max_prompt_num: # repeat prompt to ensure the same size
        #    prompt.append((cx,cy,1))
    else: # if this image doesn't have target object
        prompt = [(0,0,-1)]
        mask_curr = np.zeros_like(label_msk)
        #while len(prompt)<max_prompt_num: # repeat prompt to ensure the same size
        #    prompt.append((0,0,-1))
    prompt = np.array(prompt) 
    mask_curr = np.array(mask_curr,dtype=int)
    return prompt,mask_curr


def get_top_boxes(mask_cls,dist_thre_ratio=0.0,prompt_num=15,region_type='largest_15'):
    # Find all disconnected regions
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    #print('num of regions found', region_ids)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids+1):
        #find coordinates of points in the region
        binary_msk = np.where(label_msk==region_id, 1, 0)

        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(mask_cls)
        #print('curr mask over all mask ratio', r)
        ratio_list.append(r)
        regionid_list.append(region_id)

    if len(ratio_list)>0:
        # sort the region from largest to smallest
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]

        if region_type == 'random':
            # random select 1 region
            prompt_num = 1
            regionid_list = [random.choice(regionid_list)] # random choose 1 region
        elif region_type =='random_k':
            # random select k region 
            k_max = min(len(regionid_list),prompt_num)
            k = random.randint(1, k_max)
            # Randomly choose k values from p_list
            #print(k)
            regionid_list =  random.sample(regionid_list, k)
            
        elif region_type[:7] == 'largest':
            region_max_num = int(region_type.split('_')[-1])
            regionid_list = regionid_list[:min(region_max_num,len(regionid_list))]
            prompt_num = region_max_num
        else:
            prompt_num = min(len(regionid_list),3)
            
        prompt = []
        mask_curr = np.zeros_like(label_msk)
        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk==regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk,mask_curr)
            box = MaskToBoxSimple(binary_msk,dist_thre_ratio)
            prompt.append(box)

        while len(prompt)<prompt_num: # repeat prompt to ensure the same size
            prompt.append(box)
        prompt = np.array(prompt) 
        mask_curr = np.array(mask_curr,dtype=int)
    else:
        prompt = [[0,0,0,0]]
        mask_curr = np.zeros_like(label_msk)
        #while len(prompt)<prompt_num:
        #    prompt.append(prompt[0])
    return prompt,mask_curr
        
def MaskToBoxSimple(mask,random_thre=0.05):
    '''
    random_thre, the randomness at each side of box
    '''
    mask = mask.squeeze()
    
    y_max,x_max = mask.shape[0],mask.shape[1]
    
    #find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()
    
    y_thre = (y1-y0)*random_thre
    x_thre = (x1-x0)*random_thre
    
    x0 = max(0,x0-x_thre*random.random())
    x1 = min(x_max,x1+x_thre*random.random())
    
    y0 = max(0,y0-y_thre*random.random())
    y1 = min(y_max,y1+y_thre*random.random())
    

    return [x0,y0,x1,y1]

def min_max_normalize(tensor,p=0.01):
    p_min = torch.quantile(tensor,p)
    p_max = torch.quantile(tensor,1-p)
    tensor = torch.clamp(tensor,p_min,p_max)
    return tensor

