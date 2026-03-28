# REN: Fast and Efficient Region Encodings from Patch-Based Image Encoders
**Authors**: [Savya Khosla](https://savya08.github.io/), [Sethuraman TV](https://github.com/sethuramanio), [Barnett Lee](https://barnettlee.com/), [Alexander Schwing](https://www.alexander-schwing.de/), [Derek Hoiem](https://dhoiem.cs.illinois.edu/)

[![ArXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2505.18153)


Region Encoder Network (REN) is a lightweight model for extracting semantically meaningful region-level representations from images using point prompts. It operates on frozen patch-based vision encoder features, avoids explicit segmentation, and supports both training-free and task-specific setups across a range of vision tasks.

REN generalizes across multiple vision backbones (DINO, DINOv2, OpenCLIP) and consistently outperforms patch-based features on tasks like semantic segmentation and object retrieval. It matches the performance of SAM-based methods while being **60× faster** and using **35× less memory**.

This repo contains the PyTorch implementation and pretrained models for REN.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange.svg)



## Getting Started
Start by cloning the repo and setting up the environment.

```
git clone https://github.com/savya08/REN.git
cd REN
conda env create -f setup.yaml
conda activate ren
```

Download the region encoder checkpoints using `bash download.sh`. Alternatively, you can manually download each checkpoint to its specified save path.

| Model                 | Download                | Save Path                   |
|-----------------------|-------------------------|-----------------------------|
| REN DINO ViT-B/8      | [region encoder only](https://huggingface.co/savyak2/ren-dino-vitb8/resolve/main/checkpoint.pth)      | `logs/ren-dino-vitb8/`      |
| REN DINOv2 ViT-L/14   | [region encoder only](https://huggingface.co/savyak2/ren-dinov2-vitl14/resolve/main/checkpoint.pth)   | `logs/ren-dinov2-vitl14/`   |
| REN OpenCLIP ViT-g/14 | [region encoder only](https://huggingface.co/savyak2/ren-openclip-vitg14/resolve/main/checkpoint.pth) | `logs/ren-openclip-vitg14/` |


## Using REN
To extract region tokens from an image using REN DINOv2 ViT-L/14:

```
from ren import REN

with open('configs/ren_dinov2_vitl14.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ren = REN(config)
region_tokens = ren(<image-batch>)
```

A pretrained REN can be extended to any image encoder. E.g., to extend REN DINO ViT-B/8 to SigLIP ViT-g/16:

```
from ren import XREN

with open('configs/xren_siglip_vitg16.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
xren = XREN(config)
region_tokens = xren(image)
```

See [`test.py`](test.py) for examples of how to load REN/XREN and process images.


## Training REN
The provided REN checkpoints are trained on images from the [Ego4D dataset](https://ego4d-data.org/docs/start-here/#download-data). However, due to the large size of Ego4D, we also support training REN on the smaller [COCO dataset](https://cocodataset.org/#home). This section outlines the steps for training REN using COCO images.


### 1. Dataset Download
Download [COCO2017 train images](http://images.cocodataset.org/zips/train2017.zip), [COCO2017 val images](http://images.cocodataset.org/zips/val2017.zip), and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).


### 2. SAM 2 Download
SAM 2 masks are used to guide the training losses. Specifically, we use [SAM 2.1 Hiera Large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt).


### 3. Setup Config
To train REN with DINOv2 ViT-L/14, use [`configs/train_dinov2_vitl14.yaml`](configs/train_dinov2_vitl14.yaml). Make sure to update the following paths in the config:
```
# Path to COCO2017 dataset
coco_train_images_dir: '/path/to/coco2017/train2017/'
coco_val_images_dir: '/path/to/coco2017/val2017/'
coco_train_annotations_path: '/path/to/coco2017/annotations/instances_train2017.json'
coco_val_annotations_path: '/path/to/coco2017/annotations/instances_val2017.json'

# Path to save preprocessed data
coco_regions_rle_cache_dir: '/path/to/save/coco_region_rles/'
coco_regions_binary_cache_dir: '/path/to/save/coco_region_binaries/'
are_coco_rles_cached: False

# Path to SAM 2 checkpoint
sam2_hieral_ckpt: '/path/to/sam2.1_hiera_large.pt'
```
On the first run, SAM 2 will be used to extract region masks, which will be cached at `coco_regions_rle_cache_dir` as RLE-encoded masks. We further preprocess the RLEs into binary format to avoid decoding overhead and enable faster I/O during training. The binary masks are saved at `coco_regions_binary_cache_dir`. If you're running training a second time, set `are_coco_rles_cached` to true to reuse the cached masks.


### 4. Start Training
To start training use
```
python train.py --feature_extractor dinov2_vitl14
```
The checkpoint is saved at `logs/ren-dinov2-vitl14/checkpoint.pth`, as specified by the logging configuration in `configs/train_dinov2_vitl14.yaml`.

Note: The `--feature_extractor` argument must match the name of the corresponding YAML file in `configs/`, i.e., `train_<feature_extractor>.yaml`.

To add support for a new image encoder, update the `FeatureExtractor` class in [`model.py`](https://github.com/savya08/REN/blob/aee7645608dba43a16241ad081a991e5b376d66d/model.py#L16) with the corresponding feature extraction logic, and add a corresponding config file to `configs/`.


## License
This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.


## Citing REN
```
@inproceedings{khosla2025ren,
      title={REN: Fast and Efficient Region Encodings from Patch-Based Image Encoders}, 
      author={Savya Khosla and Sethuraman TV and Barnett Lee and Alexander Schwing and Derek Hoiem},
      booktitle={Neural Information Processing Systems},
      year={2025},
}
```
