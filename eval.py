import os
import sys
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
from fast_slic import Slic
from scipy import ndimage as ndi
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
from dataloader import VOCSegmentation, ADESegmentation
from models import VOCDecoderLinear, ADEDecoderLinear

sys.path.append('..')
sys.path.append('../segment_anything/')
from model import FeatureExtractor, RegionEncoder, TokenAggregator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def intersect_and_union(prediction, label, num_labels, ignore_index, label_map=None, reduce_labels=False,
                        reduce_predictions=False):
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    prediction = np.array(prediction)
    label = np.array(label)

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    if reduce_predictions:
        prediction[prediction == 0] = 255
        prediction = prediction - 1
        prediction[prediction == 254] = 255

    prediction = prediction[label != ignore_index]

    label = label[label!= ignore_index]
    intersect = prediction[prediction == label]
    area_intersect = np.histogram(intersect, bins=num_labels, range=(0, num_labels - 1))[0]
    area_pred_label = np.histogram(prediction, bins=num_labels, range=(0, num_labels - 1))[0]
    area_label = np.histogram(label, bins=num_labels, range=(0, num_labels - 1))[0]
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(predictions, targets, num_labels, ignore_index, label_map=None, reduce_labels=False,
                              reduce_pred_labels=False):
    total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
    total_area_union = np.zeros((num_labels,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
    total_area_label = np.zeros((num_labels,), dtype=np.float64)
    for prediction, target in tqdm(zip(predictions, targets), total=len(predictions), desc='Computing metrics'):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(prediction, target, num_labels, 
                                                                                      ignore_index, label_map, reduce_labels, 
                                                                                      reduce_pred_labels)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def mean_iou(predictions, targets, num_labels, ignore_index, nan_to_num=None, label_map=None, reduce_labels=False,
             reduce_pred_labels=False):
    total_area_intersect, total_area_union, _, total_area_label = total_intersect_and_union(predictions, targets, num_labels,
                                                                                            ignore_index, label_map, reduce_labels,
                                                                                            reduce_pred_labels)

    metrics = {}
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label

    metrics['mean_iou'] = np.nanmean(iou)
    metrics['mean_accuracy'] = np.nanmean(acc)
    metrics['overall_accuracy'] = all_acc
    metrics['per_category_iou'] = iou
    metrics['per_category_accuracy'] = acc
    if nan_to_num is not None:
        metrics = dict({metric: np.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in metrics.items()})
    return metrics


def get_slic_points(images, num_segments):
    prompts, superpixels = [], []
    for image in images:
        image = (image.permute(1, 2, 0).cpu().numpy().copy() * 255).astype(np.uint8)

        # Get SLIC superpixels
        slic = Slic(num_components=num_segments, compactness=256)
        segments = slic.iterate(image)
        slic_segments = segments.max() + 1

        # Get center of mass for all segments
        centers = np.array(ndi.center_of_mass(np.ones_like(segments), labels=segments, index=np.arange(slic_segments)))
        centers = np.round(centers).astype(int)

        centers[:, 0] = np.clip(centers[:, 0], 0, segments.shape[0] - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, segments.shape[1] - 1)

        # Check which centers are outside their own superpixel
        valid = segments[centers[:, 0], centers[:, 1]] == np.arange(slic_segments)

        # For invalid centers, pick a pixel inside the superpixel
        if not np.all(valid):
            for seg_id in np.where(~valid)[0]:
                mask = (segments == seg_id)
                yx = np.argwhere(mask)
                if len(yx) > 0:
                    centers[seg_id] = yx[len(yx) // 2]
        centers = torch.tensor(centers, dtype=torch.int64)

        # Pad if needed
        pad_len = num_segments - len(centers)
        if pad_len > 0:
            center_padding = torch.stack([centers[-1]] * pad_len)
            centers = torch.cat([centers, center_padding], dim=0)

        prompts.append(centers)
        superpixels.append(segments)
    return prompts, superpixels


class Evaluator():
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f'Configs: {config}')

        # Instantiate the dataloaders
        self.target_data = config['data']['target_data']
        if self.target_data == 'pascal_voc':
            dataset = VOCSegmentation(config, image_set='trainval', augment=False)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_subset, val_subset = random_split(dataset, [train_size, val_size])
            self.train_loader = DataLoader(train_subset, batch_size=1, num_workers=config['parameters']['num_workers'],
                                           shuffle=True, pin_memory=True)
            self.val_loader = DataLoader(val_subset, batch_size=1, num_workers=config['parameters']['num_workers'],
                                         pin_memory=True)
            self.num_classes = config['data']['voc_num_classes']
        elif self.target_data == 'ade20k':
            train_dataset = ADESegmentation(config, split='training', augment=False)
            val_dataset = ADESegmentation(config, split='validation', augment=False)
            self.train_loader = DataLoader(train_dataset, batch_size=1, num_workers=config['parameters']['num_workers'],
                                           shuffle=True, pin_memory=True)
            self.val_loader = DataLoader(val_dataset, batch_size=1, num_workers=config['parameters']['num_workers'],
                                         pin_memory=True)
            self.num_classes = config['data']['ade_num_classes']

        # Create the models
        self.extractor_name = config['ren']['pretrained']['feature_extractors'][0]
        self.patch_size = config['ren']['pretrained']['patch_sizes'][0]
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device).eval()
        self.token_aggregator = TokenAggregator(config['ren'])
        if self.target_data == 'pascal_voc':
            self.decoder = VOCDecoderLinear(config).to(device).eval()
        elif self.target_data == 'ade20k':
            self.decoder = ADEDecoderLinear(config).to(device).eval()

        # Create prompts for region encoder
        self.image_resolution = config['ren']['parameters']['image_resolution']
        self.grid_size = self.image_resolution // self.patch_size
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Load checkpoints
        self.ren_checkpoint = os.path.join(config['ren']['logging']['save_dir'], config['ren']['logging']['exp_name'], 'checkpoint.pth')
        self.decoder_checkpoint = os.path.join(self.exp_dir, 'checkpoint.pth')
        self.load_ren()
        self.load_decoder()

        # Add colormap for visualizing results
        self.voc_colormap = np.array([
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
            (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), 
            (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)
        ], dtype=np.uint8)
        self.ade_colormap = np.random.randint(0, 256, size=(self.num_classes + 1, 3), dtype=np.uint8)

    def visualize(self, image, prediction, target, save_path, ignore_index=255):
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        if self.target_data == 'pascal_voc':
            prediction[prediction == ignore_index] = 21
            target[target == ignore_index] = 21
            prediction = self.voc_colormap[prediction]
            target = self.voc_colormap[target]
        elif self.target_data == 'ade20k':
            prediction[prediction == ignore_index] = 150
            target[target == ignore_index] = 150
            prediction = self.ade_colormap[prediction]
            target = self.ade_colormap[target]

        _, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].imshow(prediction)
        axes[1].axis('off')
        axes[2].imshow(target)
        axes[2].axis('off')
        plt.savefig(save_path)
        plt.clf()
        plt.close()

    def load_ren(self):
        if os.path.exists(self.ren_checkpoint):
            checkpoint = torch.load(self.ren_checkpoint)
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            ren_epoch = checkpoint['epoch']
            ren_iter = checkpoint['iter_count']
            print(f'Loaded REN checkpoint trained for {ren_epoch} epochs, {ren_iter} iterations.')
        else:
            print('No REN checkpoint found, exiting.')
            exit()

    def load_decoder(self):
        if os.path.exists(self.decoder_checkpoint):
            checkpoint = torch.load(self.decoder_checkpoint)
            self.decoder.load_state_dict(checkpoint['decoder_state'])
            print(f'Decoder checkpoint loaded.')
        else:
            print('No decoder checkpoint found, exiting.')
            exit()

    def step(self, batch, use_slic, aggregate_tokens):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                # Compute outputs
                _, feature_maps = self.feature_extractor(self.extractor_name, images, resize=False)
                if use_slic:
                    prompts, superpixels = get_slic_points(images, self.grid_size * self.grid_size)
                else:
                    prompts = [self.grid_points for _ in range(batch_size)]
                
                ren_outputs = self.region_encoder(feature_maps, prompts)
                if aggregate_tokens:
                    aggregated_outputs = self.token_aggregator(ren_outputs['pred_tokens'], ren_outputs['proj_tokens'],
                                                               ren_outputs['attn_scores'][-1], prompts)
                    region_tokens = aggregated_outputs['aggregated_pred_tokens']
                    grouped_points = aggregated_outputs['all_grouped_points']
                    
                    padded_region_tokens = []
                    for batch_idx in range(batch_size):
                        num_tokens, feature_dims = region_tokens[batch_idx].shape
                        pad_length = self.grid_size * self.grid_size - num_tokens
                        pad_tokens = torch.zeros((pad_length, feature_dims), dtype=region_tokens[batch_idx].dtype,
                                                 device=region_tokens[batch_idx].device)
                        padded_region_tokens.append(torch.cat([region_tokens[batch_idx], pad_tokens], dim=0))
                    padded_region_tokens = torch.stack(padded_region_tokens)
                    padded_region_tokens = padded_region_tokens.view(batch_size, self.grid_size, self.grid_size, -1)
                    padded_outputs = self.decoder(padded_region_tokens.permute(0, 3, 1, 2))
                    padded_outputs = padded_outputs.permute(0, 2, 3, 1).view(batch_size, self.grid_size * self.grid_size, -1)

                    outputs = []
                    for batch_idx in range(batch_size):
                        output = padded_outputs[batch_idx][:region_tokens[batch_idx].shape[0]]
                        batch_points = grouped_points[batch_idx]

                        point_to_pred = {}
                        for group_idx in range(len(batch_points)):
                            for point_idx in range(len(batch_points[group_idx])):
                                point = (batch_points[group_idx][point_idx][0].item(), batch_points[group_idx][point_idx][1].item())
                                point_to_pred[point] = output[group_idx]

                        batch_output = []
                        for point_prompt in prompts[batch_idx]:
                            point = tuple(point_prompt.tolist())
                            batch_output.append(point_to_pred[point])
                        batch_output = torch.stack(batch_output)
                        outputs.append(batch_output.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1))
                    outputs = torch.stack(outputs)
                else:
                    region_tokens = ren_outputs['pred_tokens']
                    region_tokens = region_tokens.view(batch_size, self.grid_size, self.grid_size, -1)
                    outputs = self.decoder(region_tokens.permute(0, 3, 1, 2))

                # Resize the outputs to the desired dimensions
                if use_slic:
                    outputs = torch.argmax(outputs, dim=1).flatten(-2)[0].cpu()
                    predictions = superpixels[0].copy()
                    for segment_idx, segment_id in enumerate(np.unique(superpixels)):
                        predictions[superpixels[0] == segment_id] = outputs[segment_idx]
                    predictions = torch.tensor(predictions)[None]
                else:
                    resized_outputs = torch.nn.functional.interpolate(outputs, size=[self.image_resolution, self.image_resolution],
                                                                      mode='bilinear')
                    resized_outputs = resized_outputs.flatten(-2).permute(0, 2, 1).reshape(-1, outputs.shape[1])
                    predictions = torch.argmax(resized_outputs, dim=-1).view(-1, self.image_resolution, self.image_resolution)

        return {
            'images': images,
            'predictions': predictions,
            'targets': masks,
        }

    def run(self, split='val', use_slic=True, aggregate_tokens=False, visualize_predictions=True):
        if split == 'val':
            dataloader = self.val_loader
        elif split == 'train':
            dataloader = self.train_loader
        
        images, predictions, targets = [], [], []
        for batch in tqdm(dataloader, desc=f'Running eval'):
            outputs = self.step(batch, use_slic, aggregate_tokens)
            images.append(outputs['images'])
            predictions.append(outputs['predictions'])
            targets.append(outputs['targets'])
        images = torch.cat(images, dim=0)
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        metrics = mean_iou(predictions.cpu().numpy(), targets.cpu().numpy(), self.num_classes, ignore_index=255)
        print(f'mean_iou: {metrics["mean_iou"]}')

        if visualize_predictions:
            num_vis = 10
            for i in range(min(num_vis, len(predictions))):
                self.visualize(images[i].cpu().permute(1, 2, 0), predictions[i].cpu(), targets[i].cpu(),
                               os.path.join(self.exp_dir, f'vis/{i}.jpg'))


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

    evaluator = Evaluator(config)
    evaluator.run()
