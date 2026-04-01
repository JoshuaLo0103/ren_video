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
from video_dataloader import VSPWClipDataset
from decoder import VSPWDecoderLinear

sys.path.append('..')
sys.path.append('../segment_anything/')
from model import FeatureExtractor, RegionEncoder, TokenAggregator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def intersect_and_union(prediction, label, num_labels, ignore_index, label_map=None, reduce_labels=True,
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
        label[(label < 0) | (label >= num_labels)] = ignore_index

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

        train_dataset = VSPWClipDataset(
            config,
            split="train",
            augment=True,
            frames_per_video=5,
            sampling="window",
            seed=seed,
        )
        val_dataset = VSPWClipDataset(
            config,
            split="val",
            augment=False,
            frames_per_video=5,
            sampling="window",
            seed=seed,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=2,
            num_workers=config['parameters']['num_workers'],
            shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=2,
            num_workers=config['parameters']['num_workers'],
            shuffle=False, pin_memory=False
        )
        self.num_classes = 124

        # Create the models
        self.extractor_name = config['ren']['pretrained']['feature_extractors'][0]
        self.patch_size = config['ren']['pretrained']['patch_sizes'][0]
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device).eval()
        self.token_aggregator = TokenAggregator(config['ren'])

        self.decoder = VSPWDecoderLinear(config).to(device).eval()

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

    def step(self, batch, aggregate_tokens: bool):
        images = batch["image"].to(device)  # [B,T,C,H,W]
        masks = batch["mask"].to(device)

        B, T, C, H, W = images.shape
        N0 = self.grid_size * self.grid_size  

        # normalize mask shape to [B,T,H,W]
        if masks.ndim == 5:  
            masks = masks.squeeze(2)

        # Flatten frames for REN/feature extractor
        images_flat = images.view(B * T, C, H, W)

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                _, feature_maps = self.feature_extractor(self.extractor_name, images_flat, resize=False)

                #prompts_flat = [self.grid_points for _ in range(B * T)]  

                #ren = self.region_encoder(feature_maps, prompts_flat)

                if aggregate_tokens:
                    pred_bt = ren["pred_tokens"]
                    proj_bt = ren["proj_tokens"]
                    attn_bt = ren["attn_scores"][-1]

                    # reshape to per-video
                    D = pred_bt.shape[-1]
                    pred = pred_bt.view(B, T, N0, D).reshape(B, T * N0, D)              
                    proj = proj_bt.view(B, T, N0, proj_bt.shape[-1]).reshape(B, T * N0, -1)

                    # build block-diagonal attention [B, T*N0, T*N0]
                    attn_bt = ren["attn_scores"][-1]  

                    # --- handle multi-head attention ---
                    if attn_bt.ndim == 4:
                        if attn_bt.shape[0] == B*T and attn_bt.shape[2] == N0 and attn_bt.shape[3] == N0:
                            attn_bt = attn_bt.mean(dim=1)
                        elif attn_bt.shape[0] == B*T and attn_bt.shape[1] == N0 and attn_bt.shape[2] == N0:
                            attn_bt = attn_bt.mean(dim=3) 
                        else:
                            raise RuntimeError(f"Unexpected attn_bt shape: {attn_bt.shape}")

                    elif attn_bt.ndim == 3:
                        pass
                    else:
                        raise RuntimeError(f"Unexpected attn_bt ndim: {attn_bt.ndim}, shape: {attn_bt.shape}")

                    attn_frame = attn_bt.view(B, T, N0, N0)
                    attn = torch.zeros((B, T * N0, T * N0), device=pred.device, dtype=attn_frame.dtype)
                    for t in range(T):
                        s, e = t*N0, (t+1)*N0
                        attn[:, s:e, s:e] = attn_frame[:, t]

                    # build time-aware grid points: list length B, each tensor [T*N0, 3] = (t, y, x)
                    grid_points_video = []
                    gp = self.grid_points.to(pred.device)  # [N0,2] (y,x) pixel coords (patch centers)
                    for b_idx in range(B):
                        pts = []
                        for t_idx in range(T):
                            tp = torch.full((N0, 1), t_idx, device=pred.device, dtype=gp.dtype)  # [N0,1]
                            pts.append(torch.cat([tp, gp], dim=1))  # [N0,3]
                        grid_points_video.append(torch.cat(pts, dim=0))  # [T*N0,3]

                    # 4) Aggregate within each video
                    agg = self.token_aggregator(pred, proj, attn, grid_points_video)
                    region_tokens = agg["aggregated_pred_tokens"]  # list of len B, each [G, D]
                    grouped_points = agg["all_grouped_points"]     # list of len B, each list of groups, each [Mi,3]
                    total_tokens_in_batch = sum([t.shape[0] for t in region_tokens])
                    total_tokens_in_video = total_tokens_in_batch / (B)
                    # 5) Pad/truncate aggregated tokens to N0 so we can batch the decoder call
                    padded_region_tokens = []
                    for b_idx in range(B):
                        tok = region_tokens[b_idx]  # [G,D]
                        G = tok.shape[0]
                        if G >= N0:
                            padded = tok[:N0]
                        else:
                            pad = torch.zeros((N0 - G, D), dtype=tok.dtype, device=tok.device)
                            padded = torch.cat([tok, pad], dim=0)
                        padded_region_tokens.append(padded)

                    padded_region_tokens = torch.stack(padded_region_tokens, dim=0)              # [B, N0, D]
                    padded_region_tokens = padded_region_tokens.view(B, self.grid_size, self.grid_size, D)

                    logits_grid = self.decoder(padded_region_tokens.permute(0, 3, 1, 2))          # [B, K, GS, GS]
                    K = logits_grid.shape[1]
                    logits_groups = logits_grid.permute(0, 2, 3, 1).reshape(B, N0, K)              # [B, N0, K]

                    outputs_grid = torch.zeros((B, T, N0, K), device=logits_groups.device, dtype=logits_groups.dtype)

                    for b_idx in range(B):
                        G = region_tokens[b_idx].shape[0]
                        group_logits = logits_groups[b_idx, :min(G, N0)]  
                        batch_groups = grouped_points[b_idx]             

                        for g_idx, members in enumerate(batch_groups):
                            if g_idx >= group_logits.shape[0]:
                                break
                            gl = group_logits[g_idx]  # [K]
                            for p in members:
                                # p: (t, y, x)
                                t_idx = int(p[0].item())
                                y = int(p[1].item())
                                x = int(p[2].item())

                                gy = y // self.patch_size
                                gx = x // self.patch_size
                                if 0 <= t_idx < T and 0 <= gy < self.grid_size and 0 <= gx < self.grid_size:
                                    p_idx = gy * self.grid_size + gx
                                    outputs_grid[b_idx, t_idx, p_idx] = gl

                    outputs = outputs_grid.view(B * T, N0, K).view(B * T, self.grid_size, self.grid_size, K)
                    outputs = outputs.permute(0, 3, 1, 2)  # [BT, K, GS, GS]

                else:
                    #pred_bt = ren["pred_tokens"]
                    #D = pred_bt.shape[-1]
                    #region_tokens = pred_bt.view(B * T, self.grid_size, self.grid_size, D)
                    outputs = self.decoder(feature_maps)  # [BT, K, GS, GS]

                logits_up = torch.nn.functional.interpolate(outputs, size=(H, W), mode="bilinear", align_corners=False)
                preds = torch.argmax(logits_up, dim=1).view(B, T, H, W)

        return {
            "images": images,
            "predictions": preds,
            "targets": masks,
            #"tokens_in_video": total_tokens_in_video,
            #"original_tokens": B * T * N0
        }

    def run(self, split="val", aggregate_tokens=True, max_batches=None, num_passes=1, start_epoch=0):
        dataloader = self.val_loader if split == "val" else self.train_loader
        dataset = dataloader.dataset

        all_mious = []
        all_tokens = []
        total_tokens = 0
        for p in range(num_passes):
            # IMPORTANT: change sampling each pass
            
            dataset.set_epoch(start_epoch + p)

            preds_all, targs_all = [], []
            tokens = 0
            count = 0
            for i, batch in enumerate(tqdm(dataloader, desc=f"Eval pass {p+1}/{num_passes}")):
                out = self.step(batch, aggregate_tokens=aggregate_tokens)
                preds_all.append(out["predictions"])
                targs_all.append(out["targets"])
                #tokens += out["tokens_in_video"]
                #total_tokens = out["original_tokens"]
                count += 1
                if max_batches is not None and (i + 1) >= max_batches:
                    break

            #mean_tokens = tokens/count
            #print(f"average tokens per video = {mean_tokens}")
            #all_tokens.append(mean_tokens)
            preds = torch.cat(preds_all, dim=0)
            targs = torch.cat(targs_all, dim=0)
            miou = mean_iou(preds.cpu().numpy(), targs.cpu().numpy(), self.num_classes, ignore_index=255)["mean_iou"]
            print(f"pass {p+1}: mean_iou={miou}")
            all_mious.append(miou)

        print("avg mean_iou:", float(np.mean(all_mious)))
        #print("avg_mean_tokens:", float(np.mean(all_tokens)))
        #print("total tokens: ", total_tokens)



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

    evaluator = Evaluator(config)
    evaluator.run(split="val", aggregate_tokens=False, num_passes=10, max_batches=None, start_epoch=0)

