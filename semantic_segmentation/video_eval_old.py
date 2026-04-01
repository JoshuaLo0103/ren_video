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
from decoder import VSPWDecoderLinear
import torch.nn as nn

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

        train_dataset = CamVidClipDataset(
            config,
            split="train",
            augment=False,
            frames_per_video=2,
            sampling="window",
            seed=seed,
            #events_csv="",
            #event_sampling=False,
        )
        val_dataset = CamVidClipDataset(
            config,
            split="val",
            augment=False,
            frames_per_video=2,
            sampling="window",
            seed=seed,
            #events_csv="../vspw_strict.csv",
            #events_csv="",
            #event_sampling=False,
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
        temporal_thr = config.get('ren', {}).get('parameters', {}).get('temporal_merging_threshold', 0.9)
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
        self.ren_checkpoint = os.path.join(config['ren']['logging']['save_dir'], config['ren']['logging']['exp_name'], '1.0 more finetune new vit.pth')
        self.decoder_checkpoint = os.path.join(self.exp_dir, 'camvid 1.0.pth')
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
            if 'mlp_state' in checkpoint:
                self.mlp.load_state_dict(checkpoint['mlp_state'], strict=False)
            else:
                print('Checkpoint has no mlp_state; MLP remains randomly initialized.')
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
        valid_tracks_before_postmerge_total = 0.0
        invalid_tracks_before_postmerge_total = 0.0
        valid_groups_after_postmerge_total = 0.0
        invalid_groups_after_postmerge_total = 0.0
        # Filled on temporal decode path: tracks from TemporalTokenAggregator, then groups after post-temporal merge.
        tracks_after_temporal_mean = None
        groups_after_track_aggregate_mean = None

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
                        TemporalTokenAggregator(merging_threshold=self.temporal_token_aggregator.merging_threshold)
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
                            # Temporal-only tracking on per-grid tokens (no in-frame merges).
                            # Each token is treated as its own region, with region_idx == token index.
                            if not disable_temporal_stage:
                                for b_idx in range(B):
                                    curr_pred_tokens = frame_tok[b_idx]   # [N0, D]
                                    curr_text_tokens = frame_proj[b_idx]  # [N0, D]
                                    curr_region_points = [(int(y.item()), int(x.item())) for y, x in gp]
                                    next_pred_tokens = self.mlp(curr_pred_tokens) if use_mlp_temporal else curr_pred_tokens
                                    temporal_trackers[b_idx].update(
                                        curr_pred_tokens=curr_pred_tokens,
                                        curr_text_aligned_tokens=curr_text_tokens,
                                        curr_region_masks=None,
                                        frame_id=t,
                                        frame_resolution=(self.grid_size, self.grid_size),
                                        next_pred_tokens=next_pred_tokens,
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
                                    # representative (y,x) point per stage-1 group for bookkeeping
                                    curr_region_points = []
                                    for members in curr_groups:
                                        if len(members) == 0:
                                            curr_region_points.append((0, 0))
                                        else:
                                            y = int(torch.round(members[:, 1].float().mean()).item())
                                            x = int(torch.round(members[:, 2].float().mean()).item())
                                            curr_region_points.append((y, x))

                                    # Pattern B association:
                                    # - if enabled: match using MLP(curr)->next prediction
                                    # - if disabled: match using original REN output tokens (no MLP)
                                    if use_mlp_temporal:
                                        # Correctness w.r.t. how the MLP is trained:
                                        # apply MLP per *member token* (prompt token granularity),
                                        # then average within the stage-1 group => mean(mlp(tokens)).
                                        # curr_pred_tokens is already mean(group_members), so we do NOT
                                        # do mlp(curr_pred_tokens) here.
                                        D = curr_pred_tokens.shape[-1]
                                        next_pred_tokens_list = []
                                        for members in curr_groups:  # each: [Mi, 3] = (t, y, x)
                                            if members.numel() == 0:
                                                next_pred_tokens_list.append(
                                                    torch.zeros((D,), device=curr_pred_tokens.device, dtype=curr_pred_tokens.dtype)
                                                )
                                                continue

                                            ys = members[:, 1]
                                            xs = members[:, 2]
                                            # Map patch-center pixel coords (y,x) -> token indices in [0..N0).
                                            gy = (ys // self.patch_size).clamp(0, self.grid_size - 1)
                                            gx = (xs // self.patch_size).clamp(0, self.grid_size - 1)
                                            token_idxs = (gy * self.grid_size + gx).long()

                                            member_tokens = frame_tok[b_idx, token_idxs]  # [Mi, D]
                                            pred_members = self.mlp(member_tokens)  # [Mi, D]
                                            next_pred_tokens_list.append(pred_members.mean(dim=0))  # [D]

                                        next_pred_tokens = torch.stack(next_pred_tokens_list, dim=0)  # [G, D]
                                    else:
                                        next_pred_tokens = curr_pred_tokens  # [G, D]

                                    temporal_trackers[b_idx].update(
                                        curr_pred_tokens=curr_pred_tokens,
                                        curr_text_aligned_tokens=curr_text_tokens,
                                        curr_region_masks=None,
                                        frame_id=t,
                                        frame_resolution=(self.grid_size, self.grid_size),
                                        next_pred_tokens=next_pred_tokens,
                                        curr_region_points=curr_region_points,
                                    )

                    if disable_temporal_stage:
                        if temporal_only:
                            # Temporal-only mode with temporal stage disabled degenerates to
                            # per-frame decode on the full grid (no grouping/tracking).
                            region_tokens = pred_bt.view(B * T, self.grid_size, self.grid_size, D)
                            outputs = self.decoder(region_tokens.permute(0, 3, 1, 2))  # [BT, K, GS, GS]
                            total_tokens_in_video = float(N0)
                            total_used_tokens_in_video = float(N0)
                        else:
                            # Stage-1-only ablation: decode each frame's groups directly.
                            outputs_grid = None
                            total_tokens_in_video = 0.0
                            total_used_tokens_in_video = 0.0
                            for b_idx in range(B):
                                for t in range(T):
                                    tok = stage1_tokens_by_frame[t][b_idx]  # [G,D]
                                    G = tok.shape[0]
                                    total_tokens_in_video += float(G)
                                    total_used_tokens_in_video += float(G)
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
                            total_tokens_in_video = total_tokens_in_video / max(B, 1)
                            total_used_tokens_in_video = total_used_tokens_in_video / max(B, 1)
                            if outputs_grid is None:
                                K = self.num_classes
                                outputs_grid = torch.zeros((B, T, N0, K), device=pred_bt.device, dtype=pred_bt.dtype)
                            outputs = outputs_grid.view(B * T, N0, K).view(B * T, self.grid_size, self.grid_size, K)
                            outputs = outputs.permute(0, 3, 1, 2)
                    else:

                        # Collect tracks, merge similar tracks (post-temporal TokenAggregator-style grouping),
                        # then decode. Uses same similarity graph as in-frame TokenAggregator (group_predictions).
                        final_region_tokens = []
                        final_grouped_points = []  # list len B; each element list[len_tracks] of tensor points [Mi,3]
                        total_tokens_in_video = 0.0
                        total_used_tokens_in_video = 0.0
                        tracks_after_temporal_sum = 0.0

                        for b_idx in range(B):
                            res = temporal_trackers[b_idx].get_result()
                            if not isinstance(res, dict):
                                final_region_tokens.append(
                                    torch.empty(0, D, device=pred_bt.device, dtype=pred_bt.dtype)
                                )
                                final_grouped_points.append([])
                                continue

                            track_pred_tokens = res["track_pred_tokens"]  # [Gf, D] pre-MLP mean (for decoding)
                            # One-step MLP vectors (last association step); same space as temporal training.
                            track_match_tokens = res.get("track_match_tokens")
                            track_members = res["track_members"]          # list len Gf of [(frame_id, region_idx), ...]
                            tracks_after_temporal_sum += float(track_pred_tokens.shape[0])
                            # Valid track/group definition: finite embedding + non-trivial norm.
                            if track_pred_tokens.numel() > 0:
                                pre_valid_mask = (
                                    torch.isfinite(track_pred_tokens).all(dim=-1)
                                    & (torch.norm(track_pred_tokens, dim=-1) > 1e-6)
                                )
                                valid_tracks_before_postmerge_total += float(pre_valid_mask.sum().item())
                                invalid_tracks_before_postmerge_total += float((~pre_valid_mask).sum().item())

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

                            # Post-temporal merge: similarity in MLP / next-step space (latest match
                            # vector per track), aligned with eval association. Fall back to raw
                            # means if match tensors missing.
                            merge_feats = (
                                track_match_tokens
                                if track_match_tokens is not None
                                and track_match_tokens.shape[0] == track_pred_tokens.shape[0]
                                else track_pred_tokens
                            )
                            if merge_feats.shape[0] > 0:
                                merge_groups = group_predictions(
                                    merge_feats,
                                    similarity_threshold=self.intra_frame_merging_threshold,
                                    min_component_size=1,
                                    merge_small_groups=False,
                                )
                                merged_pred = []
                                merged_grouped = []
                                for g in merge_groups:
                                    idx_t = torch.tensor(g, device=track_pred_tokens.device, dtype=torch.long)
                                    merged_pred.append(track_pred_tokens[idx_t].mean(dim=0))
                                    parts = [grouped_tracks[i] for i in g]
                                    if len(parts) > 0:
                                        merged_grouped.append(torch.cat(parts, dim=0))
                                    else:
                                        merged_grouped.append(
                                            torch.zeros((0, 3), device=track_pred_tokens.device, dtype=torch.int64)
                                        )
                                track_pred_tokens = torch.stack(merged_pred, dim=0)
                                grouped_tracks = merged_grouped
                            if track_pred_tokens.numel() > 0:
                                post_valid_mask = (
                                    torch.isfinite(track_pred_tokens).all(dim=-1)
                                    & (torch.norm(track_pred_tokens, dim=-1) > 1e-6)
                                )
                                valid_groups_after_postmerge_total += float(post_valid_mask.sum().item())
                                invalid_groups_after_postmerge_total += float((~post_valid_mask).sum().item())

                            final_region_tokens.append(track_pred_tokens)
                            final_grouped_points.append(grouped_tracks)

                            total_tokens_in_video += float(track_pred_tokens.shape[0])
                            total_used_tokens_in_video += float(track_pred_tokens.shape[0])

                        total_tokens_in_video = total_tokens_in_video / max(B, 1)
                        total_used_tokens_in_video = total_used_tokens_in_video / max(B, 1)
                        tracks_after_temporal_mean = tracks_after_temporal_sum / max(B, 1)
                        groups_after_track_aggregate_mean = total_tokens_in_video

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
                    total_tokens_in_video = float(N0)
                    total_used_tokens_in_video = float(N0)
                    stage1_groups_total = float(B * T * N0)
                    stage1_groups_count = B * T

                logits_up = torch.nn.functional.interpolate(outputs, size=(H, W), mode="bilinear", align_corners=False)
                preds = torch.argmax(logits_up, dim=1).view(B, T, H, W)

        dbg = {
            "stage1_groups_per_frame": stage1_groups_total / max(stage1_groups_count, 1),
            "tracks_per_video_raw": float(total_tokens_in_video),
            "tracks_per_video_used": float(total_used_tokens_in_video),
            "valid_tokens_ratio": float(valid_tokens_total / max(valid_tokens_count, 1)),
            "invalid_tokens_ratio": float(1.0 - (valid_tokens_total / max(valid_tokens_count, 1))),
            "valid_tracks_before_postmerge": float(valid_tracks_before_postmerge_total / max(B, 1)),
            "invalid_tracks_before_postmerge": float(invalid_tracks_before_postmerge_total / max(B, 1)),
            "valid_groups_after_postmerge": float(valid_groups_after_postmerge_total / max(B, 1)),
            "invalid_groups_after_postmerge": float(invalid_groups_after_postmerge_total / max(B, 1)),
        }
        if tracks_after_temporal_mean is not None:
            dbg["tracks_after_temporal"] = float(tracks_after_temporal_mean)
            dbg["groups_after_track_aggregate"] = float(groups_after_track_aggregate_mean)
            dbg["tracks_per_video_raw"] = float(tracks_after_temporal_mean)
            dbg["tracks_per_video_used"] = float(groups_after_track_aggregate_mean)

        return {
            "images": images,
            "predictions": preds,
            "targets": masks,
            "tokens_in_video": total_tokens_in_video,
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
    ):
        dataloader = self.val_loader if split == "val" else self.train_loader
        dataset = dataloader.dataset

        all_mious = []
        all_tokens = []

        for p in range(num_passes):
            dataset.set_epoch(start_epoch + p)

            confmat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device="cpu")
            tokens = 0.0
            used_tokens = 0.0
            stage1_groups = 0.0
            tracks_after_temporal = 0.0
            groups_after_track_aggregate = 0.0
            valid_token_ratio = 0.0
            valid_tracks_before_postmerge = 0.0
            invalid_tracks_before_postmerge = 0.0
            valid_groups_after_postmerge = 0.0
            invalid_groups_after_postmerge = 0.0
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

                tokens += float(out["tokens_in_video"])
                if "debug_stats" in out:
                    ds = out["debug_stats"]
                    used_tokens += float(ds["tracks_per_video_used"])
                    stage1_groups += float(ds["stage1_groups_per_frame"])
                    valid_token_ratio += float(ds.get("valid_tokens_ratio", 0.0))
                    valid_tracks_before_postmerge += float(ds.get("valid_tracks_before_postmerge", 0.0))
                    invalid_tracks_before_postmerge += float(ds.get("invalid_tracks_before_postmerge", 0.0))
                    valid_groups_after_postmerge += float(ds.get("valid_groups_after_postmerge", 0.0))
                    invalid_groups_after_postmerge += float(ds.get("invalid_groups_after_postmerge", 0.0))
                    if "tracks_after_temporal" in ds:
                        tracks_after_temporal += float(ds["tracks_after_temporal"])
                        groups_after_track_aggregate += float(ds["groups_after_track_aggregate"])
                        count_track_stats += 1
                count += 1

                if max_batches is not None and (i + 1) >= max_batches:
                    break

            mean_tokens = tokens / max(count, 1)
            mean_used_tokens = used_tokens / max(count, 1)
            mean_stage1_groups = stage1_groups / max(count, 1)
            mean_valid_token_ratio = valid_token_ratio / max(count, 1)
            mean_valid_tracks_before = valid_tracks_before_postmerge / max(count, 1)
            mean_invalid_tracks_before = invalid_tracks_before_postmerge / max(count, 1)
            mean_valid_groups_after = valid_groups_after_postmerge / max(count, 1)
            mean_invalid_groups_after = invalid_groups_after_postmerge / max(count, 1)
            miou = confmat_to_miou(confmat)

            # Ordered by pipeline stage.
            print(f"average groups per frame (after in-frame merge) = {mean_stage1_groups}")
            if count_track_stats > 0:
                mean_tracks = tracks_after_temporal / count_track_stats
                mean_groups_agg = groups_after_track_aggregate / count_track_stats
                print(f"average tracks per video (before post-merge) = {mean_tracks}")
                print(f"average valid tracks per video (before post-merge) = {mean_valid_tracks_before}")
                print(f"average invalid tracks per video (before post-merge) = {mean_invalid_tracks_before}")
                print(f"average final groups per video (after post-merge) = {mean_groups_agg}")
                print(f"average valid final groups per video (after post-merge) = {mean_valid_groups_after}")
                print(f"average invalid final groups per video (after post-merge) = {mean_invalid_groups_after}")
            else:
                print(f"average tokens per video = {mean_tokens}")
            if self.debug_temporal_stats:
                print(f"average used tokens per video (after final scatter/decode set) = {mean_used_tokens}")
                print(f"average valid token ratio (raw frame tokens) = {mean_valid_token_ratio}")
            print(f"pass {p+1}: mean_iou={miou}")

            all_tokens.append(mean_tokens)
            all_mious.append(miou)

        print("avg mean_iou:", float(np.mean(all_mious)))
        print("avg_mean_tokens:", float(np.mean(all_tokens)))



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
        temporal_only=True,
        )

