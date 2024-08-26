# segment-anything2-medical-evaluation

[![arXiv Paper](https://img.shields.io/badge/arXiv-2304.10517-orange.svg?style=flat)](https://arxiv.org/abs/2408.00756)

#### By Haoyu Dong*, Hanxue Gu*, Yaqian Chen, Jichen Yang, Yuwen Chen, and Maciej A. Mazurowski

This is the official repository for our paper: [Segment anything model 2: an application to 2D and 3D medical images](https://https://arxiv.org/abs/2408.00756), where we evaluated Meta AI's Segment Anything Model 2 (SAM2) on many medical imaging datasets. The code will be ready momentarily.

## Installation

The code requires installing SAM2's repository [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2). The model and dependencies can be found in SA2M's repository, or you can install them with

```
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```

## Getting start
First, download SAM's model checkpoint 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

Note that our evaluation is conducted on sam2_hiera_large.pt, but you can switch to other models simply by changing the checkpoint path and the model configuration. 


If you want to run SAM (and competing methods) with iterative prompts, run the code with:
```
python3 eval_sam2_3d.py --dataset DATASET_PATH --num_class NUM_OF_CLASS (--bidirectional)
```

By default, it will run with fmode=2 (selecting the middle slice), pmode=1 (1 point at the center of the **largest** component). 
More choices are included in Figure 2, [![arXiv Paper](https://img.shields.io/badge/arXiv-2304.10517-orange.svg?style=flat)](https://arxiv.org/abs/2408.00756).

## Obtaining datasets from our paper

Although all the evaluations were conducted on publicly available datasets, we do not plan to share them directly since many of them require additional agreement before use. We kindly ask you to follow the official website for each dataset. 

## Adding custom datasets
To evaluate your own dataset, you need to format the dataset as: 
```
  XXX:
     images:
        volume1:
          00000.jpg
          00001.jpg
          ...
        volume2:
          00000.jpg
          00001.jpg
          ...
     masks:
        volume1:
          00000.png
          00001.png
          ...
        volume2:
          00000.png
          00001.png
          ...
```
where images and masks should have the same name. Note that images have to be **jpg** format as required by SAM 2. You can also follow [SAM 2's instructions on dataset format](https://github.com/facebookresearch/segment-anything-2/blob/main/sav_dataset/README.md) for details.

## News
- We are planning to release the interactive mode.
- We are planning to integrate channel selection as an additional hyperparameter. 

## Citation
If you find our work to be useful for your research, please cite our paper:
```
@article{dong2024segment,
  title={Segment anything model 2: an application to 2d and 3d medical images},
  author={Dong, Haoyu and Gu, Hanxue and Chen, Yaqian and Yang, Jichen and Mazurowski, Maciej A},
  journal={arXiv preprint arXiv:2408.00756},
  year={2024}
}
```
