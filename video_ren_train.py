import os
import yaml
import argparse
import random
import math
from tqdm import tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from video_ren_dataloader import RENDataset, collate_fn, SAVClipDataset
from model import FeatureExtractor, RegionTokensGenerator, RegionEncoder
from task_utils import print_log


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 777 # new seed
use_wandb = False  # Set to True to enable wandb logging
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.set_float32_matmul_precision('high')


class Trainer:
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print_log(f'Configs: {config}', self.exp_dir)
        # Instantiate the dataloaders
        train_dataset = RENDataset(config, split='train')
        self.train_loader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                                       collate_fn=collate_fn,
                                       num_workers=config['parameters']['num_workers'], pin_memory=True, shuffle=True)
        val_dataset = RENDataset(config, split='val')
        self.val_loader = DataLoader(val_dataset, batch_size=1,
                                     collate_fn=collate_fn,
                                     num_workers=5, pin_memory=True)
        # Set training parameters
        self.num_epochs = config['parameters']['num_epochs']
        self.total_steps = self.num_epochs * len(self.train_loader)
        self.accumulation_steps = config['parameters']['accumulation_steps']
        self.warmup_steps = config['parameters']['warmup_steps']
        self.logging_steps = config['parameters']['logging_steps']
        self.max_grad_norm = config['parameters']['max_grad_norm']
        self.upsample_features = False
        self.scaler = GradScaler()
        
        self.grid_size = config["architecture"]["grid_size"]   
        pts = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pts.append((y, x))
        
        self.grid_points = pts  

        # Create the models
        self.extractor_names = config['pretrained']['feature_extractors']
        self.feature_extractor = FeatureExtractor(config, device=device)
        self.region_tokens_generator = RegionTokensGenerator(device=device)
        self.region_encoder = RegionEncoder(config).to(device)

        self.hidden_dim = config['architecture']['hidden_dim']
        self.max_prompts = config['parameters']['max_prompts']

        D = self.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        ).to(device)
        # Small init on final layer so residual starts ≈ identity
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.mul_(0.01)
        # Freeze backbone, region pooling, and REN encoder; train only the temporal MLP head.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        # RegionTokensGenerator is stateless (no nn.Module / no parameters); pooling only.

        for p in self.region_encoder.parameters():
            p.requires_grad = False
        self.region_encoder.eval()

        for param in self.mlp.parameters():
            param.requires_grad_(True)
        self.mlp.train()

        self.frames_per_video = config.get('data', {}).get('frames_per_video', 5)
        trainable_params = list(self.mlp.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=config['parameters']['learning_rate'])
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Initialize training state
        self.start_epoch = 0
        self.start_iter = 0
        self.checkpoint_path = os.path.join(self.exp_dir, 'checkpoint_latest.pth')
        self.best_val_loss = float('inf')

        # Load trained RegionEncoder from the REN checkpoint (trained by train.py).
        self.ren_checkpoint_path = os.path.join(self.exp_dir, 'checkpoint.pth')
        self.load_ren()

        # Load MLP checkpoint if it exists
        self.load_checkpoint(full_resume=config.get('full_resume', False))

    def sample_prompts(self, n: int):
        # self.grid_points is list[(y,x)] of full grid
        if n >= len(self.grid_points):
            return self.grid_points
        idx = np.random.choice(len(self.grid_points), size=n, replace=False)
        return [self.grid_points[i] for i in idx]

    def load_ren(self):
        if not os.path.exists(self.ren_checkpoint_path):
            print_log(
                f'WARNING: No trained REN checkpoint at {self.ren_checkpoint_path}. '
                'RegionEncoder will use random init weights — MLP will train on the wrong token space!',
                self.exp_dir,
            )
            return
        ckpt = torch.load(self.ren_checkpoint_path, map_location=device)
        self.region_encoder.load_state_dict(ckpt['region_encoder_state'])
        print_log(
            f'Loaded trained RegionEncoder from {self.ren_checkpoint_path} '
            f'(epoch {ckpt.get("epoch", "?")}, iter {ckpt.get("iter_count", "?")})',
            self.exp_dir,
        )

    def load_checkpoint(self, full_resume=False):
        if not os.path.exists(self.checkpoint_path):
            print_log('No checkpoint found, starting training from scratch.', self.exp_dir)
            return

        ckpt = torch.load(self.checkpoint_path, map_location=device)

        if 'mlp_state' in ckpt:
            missing, unexpected = self.mlp.load_state_dict(ckpt['mlp_state'], strict=False)
            print_log(f'Loaded MLP from checkpoint. missing={len(missing)} unexpected={len(unexpected)}', self.exp_dir)
        else:
            print_log('Checkpoint has no mlp_state; initializing MLP randomly.', self.exp_dir)

        if not full_resume:
            self.start_epoch = 0
            self.start_iter = 0
            self.best_val_loss = float('inf')
            print_log('Loaded weights only; fresh optimizer/scheduler, training from epoch 0.', self.exp_dir)
            return

        self.start_epoch = ckpt.get('epoch', 0)
        self.start_iter = ckpt.get('iter_count', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print_log(f"Full resume from epoch {self.start_epoch}, iter {self.start_iter}", self.exp_dir)

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        elif "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
    
    def save_checkpoint(self, epoch, iter_count, val_loss, path=None):
        path = path or self.checkpoint_path
        checkpoint = {
            'epoch': epoch,
            'iter_count': iter_count,
            'best_val_loss': float(self.best_val_loss),
            'mlp_state': self.mlp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(checkpoint, path)
        print_log(f'Saved checkpoint to {path} (val_loss={val_loss:.4f}, best={self.best_val_loss:.4f})', self.exp_dir)
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
    def region_aware_contrastive_loss(self, pred_tokens_v1, pred_tokens_v2, region_ids_v1, region_ids_v2, temp=0.1):
        batch_size = pred_tokens_v1.shape[0]
        loss = 0.0
        for batch_idx in range(batch_size):
            tokens = torch.cat([pred_tokens_v1[batch_idx], pred_tokens_v2[batch_idx]], dim=0)
            ids = torch.cat([region_ids_v1[batch_idx], region_ids_v2[batch_idx]], dim=0)
            
            tokens = F.normalize(tokens, p=2, dim=1)
            sim_matrix = torch.matmul(tokens, tokens.T) / temp
            sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]

            pos_mask = ids.unsqueeze(0) == ids.unsqueeze(1)
            pos_mask.fill_diagonal_(False)

            numerator = torch.exp(sim_matrix) * pos_mask
            numerator = torch.sum(numerator, dim=1)
            denominator = torch.exp(sim_matrix)
            denominator = denominator.masked_fill(torch.eye(len(tokens), device=tokens.device, dtype=torch.bool), 0)
            denominator = torch.sum(denominator, dim=1)

            valid_tokens = torch.sum(pos_mask, dim=1) > 0
            batch_losses = -torch.log(numerator[valid_tokens] / denominator[valid_tokens])
            batch_loss = torch.mean(batch_losses)
            loss += batch_loss
        return loss / batch_size
    
    def attention_supervision_loss(self, attn_scores_v1, attn_scores_v2, regions_v1, regions_v2, loss_mask_v1, loss_mask_v2):
        num_heads = attn_scores_v1[0].shape[1]
        masks_v1 = torch.stack(regions_v1, dim=0).flatten(-2)
        masks_v2 = torch.stack(regions_v2, dim=0).flatten(-2)
        loss = 0.0

        def normalize(x, mode='sigmoid'):
            if mode == 'sigmoid':
                return F.sigmoid(x)
            elif mode == 'softmax':
                x = F.softmax(x, dim=-1)
                x_min = x.min(dim=-1, keepdim=True)[0]
                x_max = x.max(dim=-1, keepdim=True)[0]
                return (x - x_min) / (x_max - x_min + 1e-9)

        layers = [-1]
        for layer_idx in layers:
            for head in range(num_heads):
                bce_loss_a = F.binary_cross_entropy_with_logits(attn_scores_v1[layer_idx][:, head], masks_v1.float(),
                                                                reduction='none')
                bce_loss_a = (bce_loss_a.mean(dim=-1) * loss_mask_v1).sum() / loss_mask_v1.sum()
                attn_scores_a = normalize(attn_scores_v1[layer_idx][:, head], mode='sigmoid')
                intersection_a = (attn_scores_a * masks_v1).sum(dim=-1)
                union_a = attn_scores_a.sum(dim=-1) + masks_v1.sum(dim=-1)
                dice_score_a = (2 * intersection_a + 1e-6) / (union_a + 1e-6)
                dice_loss_a = 1 - (dice_score_a * loss_mask_v1).sum() / loss_mask_v1.sum()
                loss_a = bce_loss_a + dice_loss_a

                bce_loss_b = F.binary_cross_entropy_with_logits(attn_scores_v2[layer_idx][:, head], masks_v2.float(),
                                                                reduction='none')
                bce_loss_b = (bce_loss_b.mean(dim=-1) * loss_mask_v2).sum() / loss_mask_v2.sum()
                attn_scores_b = normalize(attn_scores_v2[layer_idx][:, head], mode='softmax')
                intersection_b = (attn_scores_b * masks_v2).sum(dim=-1)
                union_b = attn_scores_b.sum(dim=-1) + masks_v2.sum(dim=-1)
                dice_score_b = (2 * intersection_b + 1e-6) / (union_b + 1e-6)
                dice_loss_b = 1 - (dice_score_b * loss_mask_v2).sum() / loss_mask_v2.sum()
                loss_b = bce_loss_b + dice_loss_b

                loss += (loss_a + loss_b) / 2
        return loss / (num_heads * len(layers))

    def stack_bt(x):
    # x can be Tensor already or list[Tensor]
        if torch.is_tensor(x):
            return x
        if isinstance(x, list):
            # assume each element is [P,D]
            return torch.stack(x, dim=0)
        raise TypeError(type(x))

    def feature_similarity_loss(self, pred_tokens_v1, pred_tokens_v2, targets_v1, targets_v2, loss_mask_v1, loss_mask_v2):
        eps = 1e-8
        cos_loss_v1 = 1 - F.cosine_similarity(pred_tokens_v1, targets_v1, dim=-1, eps=eps)
        cos_loss_v1 = (cos_loss_v1 * loss_mask_v1).sum() / loss_mask_v1.sum().clamp(min=1e-6)
        cos_loss_v2 = 1 - F.cosine_similarity(pred_tokens_v2, targets_v2, dim=-1, eps=eps)
        cos_loss_v2 = (cos_loss_v2 * loss_mask_v2).sum() / loss_mask_v2.sum().clamp(min=1e-6)
        cos_loss = cos_loss_v1 + cos_loss_v2

        hidden_dim = pred_tokens_v1.shape[-1]
        pred_a = F.normalize(pred_tokens_v1.view(-1, hidden_dim), p=2, dim=-1, eps=eps)
        pred_b = F.normalize(pred_tokens_v2.view(-1, hidden_dim), p=2, dim=-1, eps=eps)
        tgt_a = F.normalize(targets_v1.view(-1, hidden_dim), p=2, dim=-1, eps=eps)
        tgt_b = F.normalize(targets_v2.view(-1, hidden_dim), p=2, dim=-1, eps=eps)
        pred_sim = torch.matmul(pred_a, pred_b.T)
        tgt_sim = torch.matmul(tgt_a, tgt_b.T)
        loss_mask = torch.matmul(loss_mask_v1.view(-1, 1).float(), loss_mask_v2.view(-1, 1).float().T)
        sim_loss = F.l1_loss(pred_sim, tgt_sim, reduction='none')
        sim_loss = (sim_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1e-6)
        out = (cos_loss + sim_loss) / 2
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    def step(self, batch, extractor_name):
        v1, v2 = batch

        v1_images = v1["images"].to(device)      # [B,T,C,H,W]
        v2_images = v2["images"].to(device)

        v1_regions = v1["regions"].to(device)    # [B,T,P,h,w]
        v2_regions = v2["regions"].to(device)

        v1_region_ids = v1["region_ids"].to(device)  # [B,T,P]
        v2_region_ids = v2["region_ids"].to(device)

        v1_loss_mask = v1["loss_mask"].to(device).float()  # [B,T,P]
        v2_loss_mask = v2["loss_mask"].to(device).float()
        v1_grid_points = v1["grid_points"]
        v2_grid_points = v2["grid_points"]

        B, T, C, H, W = v1_images.shape
        P = v1_region_ids.shape[-1]
        # Flatten time into batch
        v1_images_bt = v1_images.view(B*T, C, H, W)
        v2_images_bt = v2_images.view(B*T, C, H, W)

        v1_regions_bt = v1_regions.view(B*T, *v1_regions.shape[2:])  # [BT,P,h,w]
        v2_regions_bt = v2_regions.view(B*T, *v2_regions.shape[2:])

        v1_ids_bt = v1_region_ids.view(B*T, P)
        v2_ids_bt = v2_region_ids.view(B*T, P)
        v1_lm_bt  = v1_loss_mask.view(B*T, P)
        v2_lm_bt  = v2_loss_mask.view(B*T, P)
        v1_grid_bt = v1_grid_points[:, None, :, :].expand(B, T, P, 2).reshape(B*T, P, 2)
        v2_grid_bt = v2_grid_points[:, None, :, :].expand(B, T, P, 2).reshape(B*T, P, 2)
        
        # ===== frozen backbone forward =====
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                _, feats_v1 = self.feature_extractor(extractor_name, v1_images_bt, resize=self.upsample_features)
                _, feats_v2 = self.feature_extractor(extractor_name, v2_images_bt, resize=self.upsample_features)
                Hf, Wf = feats_v1.shape[-2], feats_v1.shape[-1]
                v1_regions_bt = v1_regions_bt.float()
                v2_regions_bt = v2_regions_bt.float()

                v1_regions_bt = F.interpolate(v1_regions_bt, size=(Hf, Wf), mode="nearest")  # masks
                v2_regions_bt = F.interpolate(v2_regions_bt, size=(Hf, Wf), mode="nearest")
                # region tokens from masks
                region_tokens_v1 = self.region_tokens_generator(feats_v1, v1_regions_bt)  # [BT,P,D]
                region_tokens_v2 = self.region_tokens_generator(feats_v2, v2_regions_bt)  # [BT,P,D]
                region_tokens_v1 = torch.stack(region_tokens_v1, dim=0)
                region_tokens_v2 = torch.stack(region_tokens_v2, dim=0)

                outputs_v1 = self.region_encoder(feats_v1, v1_grid_bt)
                outputs_v2 = self.region_encoder(feats_v2, v2_grid_bt)

                pred_v1 = outputs_v1["pred_tokens"]     # [BT,P,D]
                pred_v2 = outputs_v2["pred_tokens"]
                proj_v1 = outputs_v1["proj_tokens"]
                proj_v2 = outputs_v2["proj_tokens"]

                BT, P_check, D = pred_v1.shape
                assert BT == B * T
                assert P_check == P
                BT2, P_check2, D2 = region_tokens_v1.shape
                assert BT2 == B * T
                assert P_check2 == P
                assert D2 == D, f"Token dim mismatch: pred D={D}, region_tokens D={D2}"

                loss_cont = self.region_aware_contrastive_loss(pred_v1, pred_v2, v1_ids_bt, v2_ids_bt)
                loss_feat = self.feature_similarity_loss(
                    proj_v1, proj_v2,
                    region_tokens_v1, region_tokens_v2,
                    v1_lm_bt, v2_lm_bt
                )
        # MLP-only backward: encoder outputs are detached; optimize loss_temp only.
        pred_seq = pred_v1.view(B, T, P, D)
        with autocast(dtype=torch.bfloat16):
            eps = 1e-8
            loss_temp = torch.zeros((), device=pred_seq.device, dtype=pred_seq.dtype)
            count = 0
            for t in range(T - 1):
                ctx = pred_seq[:, t]
                tgt = pred_seq[:, t + 1]
                pred_next = ctx + self.mlp(ctx)
                per_pos = 1.0 - F.cosine_similarity(pred_next, tgt, dim=-1, eps=eps)
                valid = (v1_loss_mask[:, t] * v1_loss_mask[:, t + 1]).float()
                denom = valid.sum().clamp(min=1e-6)
                if valid.sum() < 0.5:
                    continue
                loss_temp = loss_temp + (per_pos * valid).sum() / denom
                count += 1
            if count > 0:
                loss_temp = loss_temp / count

            # If no valid (t, t+1) pairs, loss_temp is a plain zero scalar (no graph).
            # GradScaler.backward() needs a tensor tied to trainable params.
            if count > 0:
                loss = loss_temp
            else:
                loss = sum((p * 0.0).sum() for p in self.mlp.parameters())

        return {
            "loss_cont": loss_cont,
            "loss_feat": loss_feat,
            "loss_temp": loss_temp,
            "loss": loss,
        }
    
    def validate(self, extractor_name, num_batches=50):
        loss_cont, loss_feat, loss_temp, loss = 0.0, 0.0, 0.0, 0.0
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= num_batches:
                    break

                outputs = self.step(batch, extractor_name)

                loss += outputs['loss']
                loss_cont += outputs['loss_cont']
                loss_feat += outputs['loss_feat']

                loss_temp += outputs['loss_temp']

                count += 1

        if count > 0:
            loss /= count
            loss_cont /= count
            loss_feat /= count
            loss_temp /= count
        #print(f"Validation loss: {loss:.4f}, loss_cont: {loss_cont:.4f}, loss_feat: {loss_feat:.4f}, loss_temp: {loss_temp:.4f} processed: {count:.4f}")
        return {
            'loss_cont': loss_cont,
            'loss_feat': loss_feat,
            'loss_temp': loss_temp,
            'loss': loss,
        }
    
    def train(self):
        iter_count = self.start_iter
        self.optimizer.zero_grad()
        for epoch in range(self.start_epoch, self.num_epochs):
            # Update epoch for video datasets (for dynamic frame sampling)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if hasattr(self.val_loader.dataset, 'set_epoch'):
                self.val_loader.dataset.set_epoch(epoch)
            
            self.mlp.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                # Forward pass
                if epoch == self.start_epoch and batch_idx < (self.start_iter % len(self.train_loader)):
                    continue
                extractor_name = random.choice(self.extractor_names)
                train_outputs = self.step(batch, extractor_name)
                train_loss = train_outputs['loss']
                train_loss_cont = train_outputs['loss_cont']
                train_loss_feat = train_outputs['loss_feat']
                train_loss_temp = train_outputs['loss_temp']
                if not torch.isfinite(train_loss):
                    #print("⚠️ NaN loss. Skipping.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                # Backward pass
                self.scaler.scale(train_loss / self.accumulation_steps).backward()
                
                if (iter_count + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Log progress
                if (iter_count + 1) % self.logging_steps == 0:
                    self.mlp.eval()
                    val_outputs = self.validate(extractor_name=extractor_name)
                    self.mlp.train()
                    val_loss = val_outputs['loss']
                    val_loss_cont = val_outputs['loss_cont']
                    val_loss_feat = val_outputs['loss_feat']
                    val_loss_temp = val_outputs['loss_temp']
                    latest_path = os.path.join(self.exp_dir, 'checkpoint_latest.pth')
                    self.save_checkpoint(epoch, iter_count, val_loss, path=latest_path)
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, iter_count, val_loss)
                        print(f"Validation loss: {val_loss:.4f}, loss_cont: {val_loss_cont:.4f}, loss_feat: {val_loss_feat:.4f}, loss_temp: {val_loss_temp:.4f} (new best)")
                    else:
                        print_log(
                            f"No best save: val_loss={val_loss:.4f} (best={self.best_val_loss:.4f}). Latest → {latest_path}",
                            self.exp_dir,
                        )
                        print(
                            f"Validation loss: {val_loss:.4f}, loss_cont: {val_loss_cont:.4f}, loss_feat: {val_loss_feat:.4f}, loss_temp: {val_loss_temp:.4f}"
                        )
                    if use_wandb:
                        log_dict = {
                            'train_loss': train_loss,
                            'train_loss_cont': train_loss_cont,
                            'train_loss_feat': train_loss_feat,
                            'val_loss': val_loss,
                            'val_loss_cont': val_loss_cont,
                            'val_loss_feat': val_loss_feat,
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                        }
                        if 'loss_temp' in train_outputs:
                            log_dict['train_loss_temp'] = train_outputs['loss_temp']
                        if val_loss_temp.item() > 0:
                            log_dict['val_loss_temp'] = val_loss_temp
                        wandb.log(log_dict)

                pbar.set_postfix({
                    "loss": f"{train_loss.item():.4f}",
                    "cont": f"{train_loss_cont.item():.4f}",
                    "feat": f"{train_loss_feat.item():.4f}",
                    "temp": f"{train_loss_temp.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                iter_count += 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_extractor', type=str, default='dinov2_vitl14',
                        help='Name of the feature extractor (e.g., dinov2_vitl14).')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML. If not set, uses configs/train_{feature_extractor}.yaml')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--full_resume', action='store_true',
                        help='Fully resume training from checkpoint (restore optimizer/scheduler/epoch). Without this flag, only model weights are loaded.')
    args = parser.parse_args()

    # Set wandb flag
    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project='ren')

    config_path = args.config or f'configs/train_{args.feature_extractor}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.full_resume:
        config['full_resume'] = True

    trainer = Trainer(config)
    trainer.train()