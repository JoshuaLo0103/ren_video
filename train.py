import os
import yaml
import argparse
import random
import math
from tqdm import tqdm
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from dataloader import RENDataset, collate_fn
from model import FeatureExtractor, RegionTokensGenerator, RegionEncoder
from task_utils import print_log


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
use_wandb = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.set_float32_matmul_precision('high')
if use_wandb:
    wandb.init(project='ren')


class Trainer:
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print_log(f'Configs: {config}', self.exp_dir)

        # Instantiate the dataloaders
        train_dataset = RENDataset(config, split='train')
        self.train_loader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                                       sampler=train_dataset.get_weighted_sampler(), collate_fn=collate_fn,
                                       num_workers=config['parameters']['num_workers'], pin_memory=True)
        val_dataset = RENDataset(config, split='val')
        self.val_loader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'],
                                     sampler=val_dataset.get_weighted_sampler(), collate_fn=collate_fn,
                                     num_workers=config['parameters']['num_workers'], pin_memory=True)

        # Set training parameters
        self.num_epochs = config['parameters']['num_epochs']
        self.total_steps = self.num_epochs * len(self.train_loader) // config['parameters']['batch_size']
        self.accumulation_steps = config['parameters']['accumulation_steps']
        self.warmup_steps = config['parameters']['warmup_steps']
        self.logging_steps = config['parameters']['logging_steps']
        self.max_grad_norm = config['parameters']['max_grad_norm']
        self.upsample_features = config['parameters']['upsample_features']
        self.scaler = GradScaler()
        
        # Create the models
        self.extractor_names = config['pretrained']['feature_extractors']
        self.feature_extractor = FeatureExtractor(config, device=device)
        self.region_tokens_generator = RegionTokensGenerator(device=device)
        self.region_encoder = RegionEncoder(config).to(device)

        # Define the optimizer and loss function
        self.optimizer = optim.AdamW(self.region_encoder.parameters(), lr=config['parameters']['learning_rate'])
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Initialize training state
        self.start_epoch = 0
        self.start_iter = 0
        self.checkpoint_path = os.path.join(self.exp_dir, 'checkpoint.pth')
        self.best_val_loss = float('inf')

        # Load checkpoint if it exists
        self.load_checkpoint()

    def save_checkpoint(self, epoch, iter_count, val_loss):
        checkpoint = {
            'epoch': epoch,
            'iter_count': iter_count,
            'best_val_loss': val_loss,
            'region_encoder_state': self.region_encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)
        if use_wandb:
            wandb.save(self.checkpoint_path)
        print_log(f'Saved checkpoint with val_loss {val_loss.item():.4f}', self.exp_dir)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print_log(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}', self.exp_dir)
        else:
            print_log('No checkpoint found, starting training from scratch.', self.exp_dir)
    
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

    def feature_similarity_loss(self, pred_tokens_v1, pred_tokens_v2, targets_v1, targets_v2, loss_mask_v1, loss_mask_v2):
        cos_loss_v1 = 1 - F.cosine_similarity(pred_tokens_v1, targets_v1, dim=-1)
        cos_loss_v1 = (cos_loss_v1 * loss_mask_v1).sum() / loss_mask_v1.sum()
        cos_loss_v2 = 1 - F.cosine_similarity(pred_tokens_v2, targets_v2, dim=-1)
        cos_loss_v2 = (cos_loss_v2 * loss_mask_v2).sum() / loss_mask_v2.sum()
        cos_loss = cos_loss_v1 + cos_loss_v2

        hidden_dim = pred_tokens_v1.shape[-1]
        pred_a = F.normalize(pred_tokens_v1.view(-1, hidden_dim), p=2, dim=-1)
        pred_b = F.normalize(pred_tokens_v2.view(-1, hidden_dim), p=2, dim=-1)
        tgt_a = F.normalize(targets_v1.view(-1, hidden_dim), p=2, dim=-1)
        tgt_b = F.normalize(targets_v2.view(-1, hidden_dim), p=2, dim=-1)
        pred_sim = torch.matmul(pred_a, pred_b.T)
        tgt_sim = torch.matmul(tgt_a, tgt_b.T)
        loss_mask = torch.matmul(loss_mask_v1.view(-1, 1).float(), loss_mask_v2.view(-1, 1).float().T)
        sim_loss = F.l1_loss(pred_sim, tgt_sim, reduction='none')
        sim_loss = (sim_loss * loss_mask).sum() / loss_mask.sum()
        return (cos_loss + sim_loss) / 2
    
    def step(self, batch, extractor_name):
        v1, v2 = batch
        v1['images'] = v1['images'].to(device)
        v1['regions'] = [r.to(device) for r in v1['regions']]
        v1['region_ids'] = v1['region_ids'].to(device)
        v1['loss_mask'] = v1['loss_mask'].to(device)
        v2['images'] = v2['images'].to(device)
        v2['regions'] = [r.to(device) for r in v2['regions']]
        v2['region_ids'] = v2['region_ids'].to(device)
        v2['loss_mask'] = v2['loss_mask'].to(device)
        
        with autocast(dtype=torch.bfloat16):
            # Compute outputs for v1
            _, feature_maps_v1 = self.feature_extractor(extractor_name, v1['images'], resize=self.upsample_features)
            region_tokens_v1 = self.region_tokens_generator(feature_maps_v1, v1['regions'])
            outputs_v1 = self.region_encoder(feature_maps_v1, v1['grid_points'])

            # Compute outputs for v2
            _, feature_maps_v2 = self.feature_extractor(extractor_name, v2['images'], resize=self.upsample_features)
            region_tokens_v2 = self.region_tokens_generator(feature_maps_v2, v2['regions'])
            outputs_v2 = self.region_encoder(feature_maps_v2, v2['grid_points'])

            # Compute loss
            loss_cont = self.region_aware_contrastive_loss(outputs_v1['pred_tokens'], outputs_v2['pred_tokens'],
                                                           v1['region_ids'], v2['region_ids'])
            loss_feat = self.feature_similarity_loss(outputs_v1['proj_tokens'], outputs_v2['proj_tokens'],
                                                     torch.stack(region_tokens_v1), torch.stack(region_tokens_v2),
                                                     v1['loss_mask'], v2['loss_mask'])
            loss = loss_cont + loss_feat
        return {
            'loss_cont': loss_cont,
            'loss_feat': loss_feat,
            'loss': loss,
        }
    
    def validate(self, extractor_name, num_batches=10):
        loss_cont, loss_feat, loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx == num_batches:
                    break
                outputs = self.step(batch, extractor_name)
                loss += outputs['loss']
                loss_cont += outputs['loss_cont']
                loss_feat += outputs['loss_feat']
        loss_cont /= num_batches
        loss_feat /= num_batches
        loss /= num_batches
        return {
            'loss_cont': loss_cont,
            'loss_feat': loss_feat,
            'loss': loss,
        }
    
    def train(self):
        iter_count = self.start_iter
        self.optimizer.zero_grad()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.region_encoder.train()
            for batch in tqdm(self.train_loader, desc=f'Running epoch {epoch}'):
                # Forward pass
                extractor_name = random.choice(self.extractor_names)
                train_outputs = self.step(batch, extractor_name)
                train_loss = train_outputs['loss']
                train_loss_cont = train_outputs['loss_cont']
                train_loss_feat = train_outputs['loss_feat']

                # Backward pass
                self.scaler.scale(train_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.region_encoder.parameters(), max_norm=self.max_grad_norm)
                if (iter_count + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Log progress
                if (iter_count + 1) % self.logging_steps == 0:
                    val_outputs = self.validate(extractor_name=extractor_name)
                    val_loss = val_outputs['loss']
                    val_loss_cont = val_outputs['loss_cont']
                    val_loss_feat = val_outputs['loss_feat']
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, iter_count, val_loss)
                    if use_wandb:
                        wandb.log({
                            'train_loss': train_loss,
                            'train_loss_cont': train_loss_cont,
                            'train_loss_feat': train_loss_feat,
                            'val_loss': val_loss,
                            'val_loss_cont': val_loss_cont,
                            'val_loss_feat': val_loss_feat,
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                        })
                iter_count += 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_extractor', type=str, required=True,
                        help='Name of the feature extractor (e.g., dinov2_vitl14).')
    args = parser.parse_args()

    with open(f'configs/train_{args.feature_extractor}.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()