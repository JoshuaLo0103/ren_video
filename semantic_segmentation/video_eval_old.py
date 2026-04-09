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
from video_dataloader import VSPWClipDataset, CamVidClipDataset
from decoder import VSPWDecoderLinear, CamVidDecoderLinear
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('../segment_anything/')
from model import FeatureExtractor, RegionEncoder, TokenAggregator, TemporalTokenAggregator
from task_utils import group_predictions
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
@torch.no_grad()
def update_confmat(confmat, preds, targs, num_classes, ignore_index=255):
    """
    preds, targs: [N,H,W] int64 (or any shape flattened)
    """
    preds = preds.view(-1).to(torch.int64)
    targs = targs.view(-1).to(torch.int64)

    if ignore_index is not None:
        keep = (targs != ignore_index)
        preds = preds[keep]
        targs = targs[keep]

    k = (targs >= 0) & (targs < num_classes)
    preds = preds[k]
    targs = targs[k]

    idx = targs * num_classes + preds
    confmat += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return confmat

def confmat_to_miou(confmat):
    # confmat: [C,C] on CPU
    tp = torch.diag(confmat).float()
    fp = confmat.sum(0).float() - tp
    fn = confmat.sum(1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    miou = iou.mean().item()
    return miou

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
            augment=False,
            frames_per_video=16,
            sampling="window",
            seed=seed,
            events_csv="",
            event_sampling=False,
        )
        val_dataset = VSPWClipDataset(
            config,
            split="val",
            augment=False,
            frames_per_video=16,
            sampling="window",
            seed=seed,
            #events_csv="../vspw_strict.csv",
            events_csv="",
            event_sampling=False,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=3,
            num_workers=config['parameters']['num_workers'],
            shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=3,
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
        self.intra_frame_merging_threshold = config.get('ren', {}).get('parameters', {}).get(
            'intra_frame_merging_threshold',
            config.get('ren', {}).get('parameters', {}).get('merge_similarity', 0.95),
        )
        self.debug_temporal_stats = config.get('ren', {}).get('parameters', {}).get('debug_temporal_stats', False)
        # When using temporal-only tracking (no within-frame TokenAggregator), the
        # token count is large (N0 grid cells). A low threshold encourages merging
        # to keep track count manageable and avoid truncation at decode time.
        temporal_thr = config.get('ren', {}).get('parameters', {}).get('temporal_merging_threshold', 0.85)
        self.temporal_token_aggregator = TemporalTokenAggregator(merging_threshold=temporal_thr)
        D = config['ren']['architecture']['hidden_dim']
        
        self.mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        ).to(device).eval()   
        
        self.decoder = VSPWDecoderLinear(config).to(device).eval()

        # Create prompts for region encoder
        self.image_resolution = config['ren']['parameters']['image_resolution']
        self.grid_size = self.image_resolution // self.patch_size
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Load checkpoints
        ren_dir = os.path.join(config['ren']['logging']['save_dir'], config['ren']['logging']['exp_name'])
        self.ren_checkpoint = os.path.join(ren_dir, 'checkpoint.pth')
        self.mlp_checkpoint = os.path.join(ren_dir, 'checkpoint_latest.pth')
        self.decoder_checkpoint = os.path.join(self.exp_dir, 'mlp latest only decoder.pth')
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
        debug_random_re = False
        if debug_random_re:
            print(
                '[debug] REN_DEBUG_RANDOM_REGION_ENCODER: skipping trained RegionEncoder weights; '
                'using default __init__ weights (see seed at top of video_eval_old.py).'
            )
        else:
            if not os.path.exists(self.ren_checkpoint):
                print(f'No REN checkpoint found at {self.ren_checkpoint}, exiting.')
                exit()
            checkpoint = torch.load(self.ren_checkpoint)
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            ren_epoch = checkpoint['epoch']
            ren_iter = checkpoint['iter_count']
            print(f'Loaded RegionEncoder from checkpoint.pth (epoch {ren_epoch}, iter {ren_iter}).')

        if os.path.exists(self.mlp_checkpoint):
            mlp_ckpt = torch.load(self.mlp_checkpoint)
            print(f'[debug] MLP checkpoint path: {self.mlp_checkpoint}')
            print(f'[debug] MLP checkpoint top-level keys: {list(mlp_ckpt.keys())}')
            if 'mlp_state' in mlp_ckpt:
                sd = mlp_ckpt['mlp_state']
                print(f'[debug] mlp_state has {len(sd)} tensors; sample keys: {list(sd.keys())[:3]}')
                missing, unexpected = self.mlp.load_state_dict(sd, strict=False)
                print(f'[debug] MLP load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
                if missing:
                    print(f'[debug] missing (first 8): {missing[:8]}')
                if unexpected:
                    print(f'[debug] unexpected (first 8): {unexpected[:8]}')
                k0 = next(iter(self.mlp.state_dict()))
                print(f'[debug] MLP loaded; param {k0!r} abs-mean={self.mlp.state_dict()[k0].abs().mean().item():.6f}')
                print('Loaded MLP from checkpoint_latest.pth.')
            else:
                print('[debug] No mlp_state key in checkpoint_latest.pth; MLP remains randomly initialized.')
        else:
            print(f'[debug] No MLP checkpoint file at {self.mlp_checkpoint}; MLP remains randomly initialized.')

    def load_decoder(self):
        if os.path.exists(self.decoder_checkpoint):
            checkpoint = torch.load(self.decoder_checkpoint)
            self.decoder.load_state_dict(checkpoint['decoder_state'])
            print(f'Decoder checkpoint loaded.')
        else:
            print('No decoder checkpoint found, exiting.')
            exit()

    @torch.no_grad()
    def mlp_next_frame_temporal_loss(self, batch):
        """
        Same objective as video_ren_train.Trainer.step loss_temp: from pred_tokens at t,
        residual MLP predicts t+1 as ctx + mlp(ctx); loss is mean(1 - cos(pred_next, tgt)).
        Uses this eval script's forward (resize=False, self.grid_points). Training uses the
        same resize=False (video_ren_train.upsample_features is False).
        """
        images = batch["image"].to(device)
        B, T, C, H, W = images.shape
        N0 = self.grid_size * self.grid_size
        images_flat = images.view(B * T, C, H, W)
        self.mlp.eval()
        eps = 1e-8
        with autocast(dtype=torch.bfloat16):
            _, feature_maps = self.feature_extractor(self.extractor_name, images_flat, resize=False)
            prompts_flat = [self.grid_points for _ in range(B * T)]
            ren = self.region_encoder(feature_maps, prompts_flat)
            pred_bt = ren["pred_tokens"]
            D = pred_bt.shape[-1]
            pred_seq = pred_bt.view(B, T, N0, D)
            loss_temp = torch.zeros((), device=pred_seq.device, dtype=pred_seq.dtype)
            pair_count = 0
            for t in range(T - 1):
                ctx = pred_seq[:, t]
                tgt = pred_seq[:, t + 1]
                pred_next = ctx + self.mlp(ctx)
                per_pos = 1.0 - F.cosine_similarity(pred_next, tgt, dim=-1, eps=eps)
                finite_t = torch.isfinite(ctx).all(dim=-1) & (torch.norm(ctx, dim=-1) > 1e-6)
                finite_tp1 = torch.isfinite(tgt).all(dim=-1) & (torch.norm(tgt, dim=-1) > 1e-6)
                valid = (finite_t & finite_tp1).float()
                if valid.sum() < 0.5:
                    continue
                denom = valid.sum().clamp(min=1e-6)
                loss_temp = loss_temp + (per_pos * valid).sum() / denom
                pair_count += 1
            if pair_count > 0:
                loss_temp = loss_temp / pair_count
        return float(loss_temp.item()), pair_count

    def identity_temporal_loss(self, batch):
        """Identity baseline: predict x_{t+1} = x_t (no MLP). Returns (loss, pair_count)."""
        images = batch["image"].to(device)
        B, T, C, H, W = images.shape
        N0 = self.grid_size * self.grid_size
        images_flat = images.view(B * T, C, H, W)
        eps = 1e-8
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            _, feature_maps = self.feature_extractor(self.extractor_name, images_flat, resize=False)
            prompts_flat = [self.grid_points for _ in range(B * T)]
            ren = self.region_encoder(feature_maps, prompts_flat)
            pred_bt = ren["pred_tokens"]
            D = pred_bt.shape[-1]
            pred_seq = pred_bt.view(B, T, N0, D)
            loss_id = torch.zeros((), device=pred_seq.device, dtype=pred_seq.dtype)
            pair_count = 0
            for t in range(T - 1):
                ctx = pred_seq[:, t]
                tgt = pred_seq[:, t + 1]
                per_pos = 1.0 - F.cosine_similarity(ctx, tgt, dim=-1, eps=eps)
                finite_t = torch.isfinite(ctx).all(dim=-1) & (torch.norm(ctx, dim=-1) > 1e-6)
                finite_tp1 = torch.isfinite(tgt).all(dim=-1) & (torch.norm(tgt, dim=-1) > 1e-6)
                valid = (finite_t & finite_tp1).float()
                if valid.sum() < 0.5:
                    continue
                denom = valid.sum().clamp(min=1e-6)
                loss_id = loss_id + (per_pos * valid).sum() / denom
                pair_count += 1
            if pair_count > 0:
                loss_id = loss_id / pair_count
        return float(loss_id.item()), pair_count

    
    def step(
        self,
        batch,
        aggregate_tokens: bool,
        use_mlp_temporal: bool = True,
        disable_temporal_stage: bool = False,
        temporal_only: bool = False,
    ):
        images = batch["image"].to(device)  
        masks = batch["mask"].to(device)
        B, T, C, H, W = images.shape
        N0 = self.grid_size * self.grid_size
        valid_tokens_total = 0.0
        valid_tokens_count = 0
        tracks_after_temporal_mean = None

        # Flatten frames for REN/feature extractor
        images_flat = images.view(B * T, C, H, W)

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                _, feature_maps = self.feature_extractor(self.extractor_name, images_flat, resize=False)

                prompts_flat = [self.grid_points for _ in range(B * T)]

                ren = self.region_encoder(feature_maps, prompts_flat)

                if aggregate_tokens:
                    pred_bt = ren["pred_tokens"]   # [B*T, N0, D]
                    proj_bt = ren["proj_tokens"]   # [B*T, N0, D]

                    D = pred_bt.shape[-1]
                    pred_bt = pred_bt.view(B, T, N0, D)          # [B, T, N0, D]
                    proj_bt = proj_bt.view(B, T, N0, proj_bt.shape[-1])  # [B, T, N0, D]

                    # Two modes:
                    # - temporal_only=True: skip in-frame TokenAggregator, track per-grid tokens across time.
                    # - temporal_only=False: Stage 1 (in-frame) TokenAggregator, then Stage 2 (temporal).
                    temporal_trackers = [
                        TemporalTokenAggregator(
                            merging_threshold=self.temporal_token_aggregator.merging_threshold,
                            mlp=self.mlp if use_mlp_temporal else None,
                        )
                        for _ in range(B)
                    ]
                    for tt in temporal_trackers:
                        tt.reset()
                    gp = self.grid_points.to(pred_bt.device)  # [N0,2]
                    grouped_points_by_frame = []  # used only when temporal_only=False
                    stage1_tokens_by_frame = []   # len T, each is list len B, each [G, D]
                    stage1_groups_total = 0.0
                    stage1_groups_count = 0

                    for t in range(T):
                        frame_tok = pred_bt[:, t]    # [B, N0, D]
                        frame_proj = proj_bt[:, t]   # [B, N0, D]
                        # Eval-time token validity monitor: finite + non-zero norm.
                        frame_valid = torch.isfinite(frame_tok).all(dim=-1) & (torch.norm(frame_tok, dim=-1) > 1e-6)
                        valid_tokens_total += float(frame_valid.sum().item())
                        valid_tokens_count += int(frame_valid.numel())
                        tp = torch.full((N0, 1), t, device=frame_tok.device, dtype=gp.dtype)
                        pts = torch.cat([tp, gp], dim=1)  # [N0,3]
                        if temporal_only:
                            if not disable_temporal_stage:
                                curr_region_points = [(int(y.item()), int(x.item())) for y, x in gp]
                                for b_idx in range(B):
                                    temporal_trackers[b_idx].update(
                                        curr_pred_tokens=frame_tok[b_idx],
                                        curr_text_aligned_tokens=frame_proj[b_idx],
                                        curr_region_masks=None,
                                        frame_id=t,
                                        frame_resolution=(self.grid_size, self.grid_size),
                                        curr_region_points=curr_region_points,
                                    )
                            # For bookkeeping consistency in later scatter, we don't store grouped_points_by_frame.
                            stage1_groups_total += float(B * N0)
                            stage1_groups_count += B
                            stage1_tokens_by_frame.append([frame_tok[b_idx] for b_idx in range(B)])
                        else:
                            # Stage 1 (in-frame) aggregation
                            frame_attn = torch.zeros((B, N0, N0), device=frame_tok.device, dtype=frame_tok.dtype)
                            grid_points_video = [pts for _ in range(B)]

                            agg1 = self.token_aggregator(
                                frame_tok,
                                frame_proj,
                                frame_attn,
                                grid_points_video,
                                similarity=self.intra_frame_merging_threshold,
                            )
                            grouped_points_by_frame.append(agg1["all_grouped_points"])
                            stage1_tokens_by_frame.append(agg1["aggregated_pred_tokens"])
                            for b_idx in range(B):
                                stage1_groups_total += float(agg1["aggregated_pred_tokens"][b_idx].shape[0])
                                stage1_groups_count += 1

                            if not disable_temporal_stage:
                                for b_idx in range(B):
                                    curr_pred_tokens = agg1["aggregated_pred_tokens"][b_idx]   # [G, D]
                                    curr_text_tokens = agg1["aggregated_proj_tokens"][b_idx]   # [G, D]
                                    curr_groups = agg1["all_grouped_points"][b_idx]            # list len G, each [Mi,3]
                                    curr_region_points = []
                                    for members in curr_groups:
                                        if len(members) == 0:
                                            curr_region_points.append((0, 0))
                                        else:
                                            y = int(torch.round(members[:, 1].float().mean()).item())
                                            x = int(torch.round(members[:, 2].float().mean()).item())
                                            curr_region_points.append((y, x))

                                    temporal_trackers[b_idx].update(
                                        curr_pred_tokens=curr_pred_tokens,
                                        curr_text_aligned_tokens=curr_text_tokens,
                                        curr_region_masks=None,
                                        frame_id=t,
                                        frame_resolution=(self.grid_size, self.grid_size),
                                        curr_region_points=curr_region_points,
                                    )

                    if disable_temporal_stage:
                        if temporal_only:
                            # Temporal-only mode with temporal stage disabled degenerates to
                            # per-frame decode on the full grid (no grouping/tracking).
                            region_tokens = pred_bt.view(B * T, self.grid_size, self.grid_size, D)
                            outputs = self.decoder(region_tokens.permute(0, 3, 1, 2))  # [BT, K, GS, GS]
                        else:
                            # Stage-1-only ablation: decode each frame's groups directly.
                            outputs_grid = None
                            for b_idx in range(B):
                                for t in range(T):
                                    tok = stage1_tokens_by_frame[t][b_idx]  # [G,D]
                                    G = tok.shape[0]
                                    if G == 0:
                                        continue
                                    # Decode all stage-1 groups directly: [1, D, G, 1] -> [1, K, G, 1]
                                    logits_grid = self.decoder(tok.t().unsqueeze(0).unsqueeze(-1))
                                    K = logits_grid.shape[1]
                                    logits_groups = logits_grid.squeeze(0).squeeze(-1).transpose(0, 1)  # [G, K]
                                    if outputs_grid is None:
                                        outputs_grid = torch.zeros((B, T, N0, K), device=logits_groups.device, dtype=logits_groups.dtype)

                                    batch_groups = grouped_points_by_frame[t][b_idx]
                                    group_logits = logits_groups[:len(batch_groups)]
                                    for g_idx, members in enumerate(batch_groups):
                                        if g_idx >= group_logits.shape[0]:
                                            break
                                        gl = group_logits[g_idx]
                                        for p in members:
                                            t_idx = int(p[0].item())
                                            y = int(p[1].item())
                                            x = int(p[2].item())
                                            gy = y // self.patch_size
                                            gx = x // self.patch_size
                                            if 0 <= t_idx < T and 0 <= gy < self.grid_size and 0 <= gx < self.grid_size:
                                                p_idx = gy * self.grid_size + gx
                                                outputs_grid[b_idx, t_idx, p_idx] = gl
                            if outputs_grid is None:
                                K = self.num_classes
                                outputs_grid = torch.zeros((B, T, N0, K), device=pred_bt.device, dtype=pred_bt.dtype)
                            outputs = outputs_grid.view(B * T, N0, K).view(B * T, self.grid_size, self.grid_size, K)
                            outputs = outputs.permute(0, 3, 1, 2)
                    else:

                        final_region_tokens = []
                        final_grouped_points = []
                        tracks_after_temporal_sum = 0.0

                        for b_idx in range(B):
                            res = temporal_trackers[b_idx].get_result()
                            if not isinstance(res, dict):
                                final_region_tokens.append(
                                    torch.empty(0, D, device=pred_bt.device, dtype=pred_bt.dtype)
                                )
                                final_grouped_points.append([])
                                continue

                            track_pred_tokens = res["track_pred_tokens"]  # [Gf, D]
                            track_members = res["track_members"]          # list len Gf of [(frame_id, region_idx), ...]
                            tracks_after_temporal_sum += float(track_pred_tokens.shape[0])

                            # Convert track members to original (t,y,x) member points.
                            grouped_tracks = []
                            for members_track in track_members:
                                points = []
                                for (f_id, region_idx) in members_track:
                                    if temporal_only:
                                        # region_idx is token index in [0..N0)
                                        if 0 <= f_id < T and 0 <= region_idx < N0:
                                            y = int(gp[region_idx, 0].item())
                                            x = int(gp[region_idx, 1].item())
                                            points.append(
                                                torch.tensor([[f_id, y, x]], device=track_pred_tokens.device, dtype=torch.int64)
                                            )
                                    else:
                                        if 0 <= f_id < len(grouped_points_by_frame):
                                            frame_groups_b = grouped_points_by_frame[f_id][b_idx]
                                            if 0 <= region_idx < len(frame_groups_b):
                                                points.append(frame_groups_b[region_idx].to(track_pred_tokens.device))
                                if len(points) > 0:
                                    grouped_tracks.append(torch.cat(points, dim=0))
                                else:
                                    grouped_tracks.append(torch.zeros((0, 3), device=track_pred_tokens.device, dtype=torch.int64))

                            final_region_tokens.append(track_pred_tokens)
                            final_grouped_points.append(grouped_tracks)

                        tracks_after_temporal_mean = tracks_after_temporal_sum / max(B, 1)

                        # Decode all track tokens directly, then scatter logits back.
                        outputs_grid = None
                        for b_idx in range(B):
                            tok = final_region_tokens[b_idx]  # [G, D]
                            if tok.shape[0] == 0:
                                continue
                            logits_grid = self.decoder(tok.t().unsqueeze(0).unsqueeze(-1))  # [1,K,G,1]
                            K = logits_grid.shape[1]
                            group_logits = logits_grid.squeeze(0).squeeze(-1).transpose(0, 1)  # [G, K]
                            if outputs_grid is None:
                                outputs_grid = torch.zeros((B, T, N0, K), device=group_logits.device, dtype=group_logits.dtype)

                            batch_groups = final_grouped_points[b_idx]
                            for g_idx, members in enumerate(batch_groups):
                                if g_idx >= group_logits.shape[0]:
                                    break
                                gl = group_logits[g_idx]
                                for p in members:
                                    t_idx = int(p[0].item())
                                    y = int(p[1].item())
                                    x = int(p[2].item())
                                    gy = y // self.patch_size
                                    gx = x // self.patch_size
                                    if 0 <= t_idx < T and 0 <= gy < self.grid_size and 0 <= gx < self.grid_size:
                                        p_idx = gy * self.grid_size + gx
                                        outputs_grid[b_idx, t_idx, p_idx] = gl

                        if outputs_grid is None:
                            K = self.num_classes
                            outputs_grid = torch.zeros((B, T, N0, K), device=pred_bt.device, dtype=pred_bt.dtype)

                        outputs = outputs_grid.view(B * T, N0, K).view(B * T, self.grid_size, self.grid_size, K)
                        outputs = outputs.permute(0, 3, 1, 2)

                else:
                    pred_bt = ren["pred_tokens"]
                    D = pred_bt.shape[-1]
                    region_tokens = pred_bt.view(B * T, self.grid_size, self.grid_size, D)
                    outputs = self.decoder(region_tokens.permute(0, 3, 1, 2)) 
                    stage1_groups_total = float(B * T * N0)
                    stage1_groups_count = B * T

                logits_up = torch.nn.functional.interpolate(outputs, size=(H, W), mode="bilinear", align_corners=False)
                preds = torch.argmax(logits_up, dim=1).view(B, T, H, W)

        dbg = {
            "stage1_groups_per_frame": stage1_groups_total / max(stage1_groups_count, 1),
        }
        if tracks_after_temporal_mean is not None:
            dbg["tracks_per_video"] = float(tracks_after_temporal_mean)

        return {
            "images": images,
            "predictions": preds,
            "targets": masks,
            "debug_stats": dbg,
        }

    def run(
        self,
        split="val",
        aggregate_tokens=True,
        max_batches=None,
        num_passes=1,
        start_epoch=0,
        use_mlp_temporal: bool = True,
        disable_temporal_stage: bool = False,
        temporal_only: bool = False,
        mlp_temporal_sanity_check: bool = True,
    ):
        dataloader = self.val_loader if split == "val" else self.train_loader
        dataset = dataloader.dataset

        skip_sanity = os.environ.get('REN_SKIP_MLP_TEMPORAL_SANITY', '').lower() in ('1', 'true', 'yes')
        if mlp_temporal_sanity_check and not skip_sanity:
            try:
                dataset.set_epoch(start_epoch)
                batch0 = next(iter(dataloader))
                lt, n_pairs = self.mlp_next_frame_temporal_loss(batch0)
                print(
                    f'[mlp temporal sanity] loss_temp (mean 1-cos over time, like video_ren_train): '
                    f'{lt:.6f}  (used {n_pairs} (t,t+1) steps, T={batch0["image"].shape[1]})'
                )
                li, ni = self.identity_temporal_loss(batch0)
                delta = li - lt
                print(
                    f'[identity baseline]  loss_temp: {li:.6f}  (used {ni} steps)  '
                    f'MLP improvement: {delta:+.6f} ({"better" if delta > 0 else "WORSE"} than identity)'
                )
            except StopIteration:
                print('[mlp temporal sanity] skipped (empty dataloader)')
            except Exception as e:
                print(f'[mlp temporal sanity] failed: {e}')

        all_mious = []

        for p in range(num_passes):
            dataset.set_epoch(start_epoch + p)

            confmat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device="cpu")
            stage1_groups = 0.0
            tracks_after_temporal = 0.0
            count = 0
            count_track_stats = 0

            for i, batch in enumerate(tqdm(dataloader, desc=f"Eval pass {p+1}/{num_passes}")):
                out = self.step(
                    batch,
                    aggregate_tokens=aggregate_tokens,
                    use_mlp_temporal=use_mlp_temporal,
                    disable_temporal_stage=disable_temporal_stage,
                    temporal_only=temporal_only,
                )

                preds = out["predictions"]   # may be [N,H,W] or [N,C,H,W]
                targs = out["targets"]       # may be [B,T,H,W] or [N,H,W]

                confmat = update_confmat(
                    confmat,
                    preds.detach().cpu(),
                    targs.detach().cpu(),
                    num_classes=self.num_classes,
                    ignore_index=255
                )

                if "debug_stats" in out:
                    ds = out["debug_stats"]
                    stage1_groups += float(ds["stage1_groups_per_frame"])
                    if "tracks_per_video" in ds:
                        tracks_after_temporal += float(ds["tracks_per_video"])
                        count_track_stats += 1
                count += 1

                if max_batches is not None and (i + 1) >= max_batches:
                    break

            mean_stage1_groups = stage1_groups / max(count, 1)
            miou = confmat_to_miou(confmat)

            print(f"average groups per frame (after in-frame merge) = {mean_stage1_groups}")
            if count_track_stats > 0:
                mean_tracks = tracks_after_temporal / count_track_stats
                print(f"average tracks per video = {mean_tracks}")
            print(f"pass {p+1}: mean_iou={miou}")

            all_mious.append(miou)

        print("avg mean_iou:", float(np.mean(all_mious)))



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

    evaluator = Evaluator(config)
    evaluator.run(  
        split="val",
        aggregate_tokens=True,
        num_passes=5,
        max_batches=None,
        start_epoch=0,
        disable_temporal_stage=False,
        use_mlp_temporal=False,
        temporal_only=False,
        )

