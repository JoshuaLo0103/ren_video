import os
import numpy as np
from matplotlib import pyplot as plt
from fast_slic import Slic
from scipy import ndimage as ndi
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from model import FeatureExtractor, RegionEncoder, TokenAggregator


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class REN(nn.Module):
    def __init__(self, config):
        super(REN, self).__init__()
        
        # Create the models
        self.extractor_name = config['parameters']['feature_extractor']
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device).eval()
        self.token_aggregator = TokenAggregator(config['ren'])

        # Load the checkpoint
        self.checkpoint_path = config['parameters']['ren_ckpt']
        self.load_checkpoint()
        
        # Grid points for region tokens
        image_resolution = config['parameters']['image_resolution']
        self.grid_size = config['parameters']['grid_size']
        x_coords = np.linspace(1, image_resolution - 2, self.grid_size, dtype=int)
        y_coords = np.linspace(1, image_resolution - 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Fetch mode parameters from config
        self.use_slic = config['parameters']['use_slic']
        self.aggregate_tokens = config['parameters']['aggregate_tokens']
        self.token_variant = config['parameters']['token_variant']
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            print(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}.')
        else:
            print('No checkpoint found; exiting..')
            exit()

    def get_slic_points(self, images, num_segments):
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

    def forward(self, images):
        _, feature_maps = self.feature_extractor(self.extractor_name, images, resize=False)
        if self.use_slic:
            prompts, _ = self.get_slic_points(images, self.grid_size * self.grid_size)
        else:
            prompts = [self.grid_points for _ in range(images.shape[0])]
        ren_outputs = self.region_encoder(feature_maps, prompts)
        if self.aggregate_tokens:
            aggregated_outputs = self.token_aggregator(ren_outputs['pred_tokens'], ren_outputs['proj_tokens'],
                                                       ren_outputs['attn_scores'][-1], prompts)
            if self.token_variant == 'ren':
                region_tokens = aggregated_outputs['aggregated_pred_tokens']
            elif self.token_variant == 'aligned_ren':
                region_tokens = aggregated_outputs['aggregated_proj_tokens']
            else:
                raise ValueError(f'Set `token_variant` in config to either "ren" or "aligned_ren".')
        else:
            if self.token_variant == 'ren':
                region_tokens = ren_outputs['pred_tokens']
            elif self.token_variant == 'aligned_ren':
                region_tokens = ren_outputs['proj_tokens']
            else:
                raise ValueError(f'Set `token_variant` in config to either "ren" or "aligned_ren".')
        return region_tokens
    

class XREN(nn.Module):
    def __init__(self, config):
        super(XREN, self).__init__()

        # Instantiate REN modules
        self.ren_extractor_name = config['parameters']['ren_feature_extractor']
        self.target_extractor_name = config['parameters']['target_feature_extractor']
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device).eval()
        self.token_aggregator = TokenAggregator(config['ren'])

        # Instantiate target models
        self.target_extractor_name = config['parameters']['target_feature_extractor']
        if self.target_extractor_name == 'siglip_vitg16':
            self.target_model = AutoModel.from_pretrained('google/siglip2-giant-opt-patch16-384',
                                                          device_map=device).vision_model

        # Load the checkpoint
        self.checkpoint_path = config['parameters']['ren_ckpt']
        self.load_checkpoint()

        # Grid points for region tokens
        self.image_resolution = config['parameters']['image_resolution']
        self.grid_size = config['parameters']['grid_size']
        x_coords = np.linspace(1, self.image_resolution - 2, self.grid_size, dtype=int)
        y_coords = np.linspace(1, self.image_resolution - 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            print(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}.')
        else:
            print('No checkpoint found; exiting..')
            exit()

    def get_slic_points(self, images, num_segments):
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

    def get_pooled_tokens(self, images, region_masks, batch_size=1024):
        if self.target_extractor_name == 'siglip_vitg16':
            transform = kornia.augmentation.AugmentationSequential(
                kornia.augmentation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            )
            transformed_images = transform(images)
            patch_length = 16

            for i in range(0, transformed_images.shape[0], batch_size):
                image_batch = transformed_images[i:(i + batch_size)].to(device=images.device)
                with torch.inference_mode():
                    features_out = self.target_model(pixel_values=image_batch, output_hidden_states=False,
                                                     output_attentions=False).last_hidden_state
                    pooled_tokens = []
                    for image_idx in range(features_out.shape[0]):
                        pooled_tokens.append([])
                        image_features = features_out[image_idx]
                        mask_resolution = self.image_resolution // patch_length
                        masks = F.interpolate(region_masks[image_idx].unsqueeze(1), size=(mask_resolution, mask_resolution),
                                              mode='nearest').squeeze(1)
                        for mask in masks:
                            mask_flatten = mask.view(-1)
                            sampled_image_features = image_features[mask_flatten]
                            pooled_token = self.target_model.head(sampled_image_features.unsqueeze(0))
                            pooled_tokens[-1].append(pooled_token)
                        pooled_tokens[-1] = torch.cat(pooled_tokens[-1])
        return pooled_tokens
    
    def forward(self, images, visualize_mask=False):
        _, feature_maps = self.feature_extractor(self.ren_extractor_name, images, resize=False)
        prompts, segments = self.get_slic_points(images, self.grid_size * self.grid_size)
        ren_outputs = self.region_encoder(feature_maps, prompts)
        aggregated_outputs = self.token_aggregator(ren_outputs['pred_tokens'], ren_outputs['proj_tokens'],
                                                   ren_outputs['attn_scores'][-1], prompts)
        all_grouped_points = aggregated_outputs['all_grouped_points']
        
        # Create region masks
        region_masks = []
        for batch_idx in range(len(all_grouped_points)):
            region_masks.append([])
            segment = segments[batch_idx]
            for group_idx in range(len(all_grouped_points[batch_idx])):
                group_points = all_grouped_points[batch_idx][group_idx]
                mask = np.zeros_like(segment, dtype=bool)
                for yx in group_points:
                    y, x = yx
                    sp_label = segment[y, x]
                    mask |= (segment == sp_label)
                region_masks[batch_idx].append(torch.tensor(mask.astype(np.uint8)))

                if visualize_mask:
                    plt.imshow(mask[..., None] * images[batch_idx].permute(1, 2, 0).cpu().numpy())
                    plt.savefig(f'{batch_idx}-{group_idx}.jpg')
                    plt.imshow(images[batch_idx].permute(1, 2, 0).cpu().numpy())
                    plt.savefig(f'{batch_idx}-image.jpg')
        region_masks = [torch.stack(masks).to(device) for masks in region_masks]

        # Pool features of the target model
        pooled_tokens = self.get_pooled_tokens(images, region_masks)
        return pooled_tokens