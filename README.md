# Semantic Segmentation with REN
This folder contains code and scripts for evaluating REN representations on standard semantic segmentation benchmarks such as **Pascal VOC 2012** and **ADE20K**.

REN generates region tokens for each point prompt on the input image, and a linear classifier predicts the class label for each token. Predictions are mapped back to the image by assigning the label to the superpixel containing the corresponding prompt. The results are presented below.


| Method        | VOC2012        | ADE20K         |
|---------------|----------------|----------------|
| DINOv2        | 82.1           | 47.7           |
| REN-DINOv2    | **86.5**       | **50.9**       |
| DINO          | 66.4           | 31.8           |
| REN-DINO      | **71.4**       | **35.1**       |
| OpenCLIP      | 71.4           | 39.3           |
| REN-OpenCLIP  | **78.0**       | **42.8**       |



## Prepare Datasets
Download the datasets:

- [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/)
- [ADE20K](https://ade20k.csail.mit.edu/)

Set the following in `config.yaml`:
```
target_data: 'ade20k'
voc_root_dir: '/path/to/VOCdevkit/'
ade_root_dir: '/path/to/ADEChallengeData2016/'
```


## Training
To train a linear decoder for the target data, run:
```
python train.py
```
The current config file is set up for DINOv2 ViT-L/14. To use a different image encoder, update the parameters under the `ren` scope in the config file, including `feature_extractors`, `patch_sizes`, `image_resolution`, `grid_size`, `hidden_dims`, and `exp_name`. The trained checkpoint will be saved in the `logs/` directory.


## Evaluation
To evaluate, run:
```
python eval.py
```
This uses the same `config.yaml` as training and automatically loads the checkpoint saved during training. By default, 10 prediction visualizations are generated and saved in the checkpoint directory for inspection.