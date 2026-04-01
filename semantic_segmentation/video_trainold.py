import os
import sys
import math
import yaml
import random
import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from video_dataloader import VSPWClipDataset
from decoder import VSPWDecoderLinear

sys.path.append('..')
sys.path.append('../segment_anything/')
from model import FeatureExtractor, RegionEncoder, TokenAggregator
from task_utils import group_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
use_wandb = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.set_float32_matmul_precision('high')
if use_wandb:
    wandb.init(project='ren')


class Trainer():
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f'Configs: {config}')

        # Instantiate the dataloaders
        root = config["data"]["vspw_root_dir"]
        clip_len = 3 #config["data"].get("clip_len", 5)
        resize = 224 #config["ren"]["parameters"].get("image_resolution", 518)
        train_dataset = VSPWClipDataset(root, split="train", clip_len=clip_len, resize=resize)
        val_dataset = VSPWClipDataset(root, split='train',  clip_len=clip_len, resize=resize)
        self.train_loader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                                           num_workers=config['parameters']['num_workers'], shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'],
                                         num_workers=config['parameters']['num_workers'], pin_memory=True)
        # Set training parameters
        self.num_epochs = config['parameters']['num_epochs']
        self.total_steps = self.num_epochs * len(self.train_loader)
        self.warmup_steps = config['parameters']['warmup_steps']
        self.max_grad_norm = config['parameters']['max_grad_norm']
        self.accumulation_steps = config['parameters']['accumulation_steps']
        self.logging_steps = config['parameters']['logging_steps']
        self.scaler = GradScaler()
        self.merge_similarity = 0.95
        # Create the models
        self.extractor_name = config['ren']['pretrained']['feature_extractors'][0]
        self.patch_size = config['ren']['pretrained']['patch_sizes'][0]
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device)
        self.decoder = VSPWDecoderLinear(config).to(device)

        

        # Create prompts for region encoder
        self.image_resolution = config['ren']['parameters']['image_resolution']
        self.grid_size = config['ren']['architecture']['grid_size']
        x_coords = np.linspace(1, self.image_resolution - 2, self.grid_size, dtype=int)
        y_coords = np.linspace(1, self.image_resolution - 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Define the optimizer and loss function
        
        self.optimizer = optim.AdamW(self.decoder.parameters(), lr=config['parameters']['learning_rate'])
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Initialize training state
        self.start_epoch = 0
        self.start_iter = 0
        self.best_val_loss = float('inf')

        # Load checkpoints
        self.ren_checkpoint = os.path.join(config['ren']['logging']['save_dir'], config['ren']['logging']['exp_name'], 'checkpoint.pth')
        self.decoder_checkpoint = os.path.join(self.exp_dir, 'checkpoint.pth')
        self.load_ren()
        # freeze ren for now
        self.feature_extractor.eval()
        self.region_encoder.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        for p in self.region_encoder.parameters():
            p.requires_grad_(False)
        self.load_decoder()

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
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            self.decoder.load_state_dict(checkpoint['decoder_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}')
        else:
            print('No decoder checkpoint found, starting training from scratch.')

    def save_decoder(self, epoch, iter_count, val_loss):
        checkpoint = {
            'epoch': epoch,
            'iter_count': iter_count,
            'best_val_loss': val_loss,
            'decoder_state': self.decoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(checkpoint, self.decoder_checkpoint)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    def step(self, batch):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        batch_size, T, C, H, W = images.shape

        if masks.ndim == 5:  # [B,T,1,H,W]
            masks = masks.squeeze(2)

        images_flat = images.view(batch_size * T, C, H, W)
        masks_flat  = masks.view(batch_size*T, H, W)

        with autocast(dtype=torch.bfloat16):
            # Compute outputs
            with torch.no_grad():
                _, feats = self.feature_extractor(self.extractor_name, images_flat, resize=False)
                prompts = [self.grid_points for _ in range(batch_size * T)]
                pred_tokens = self.region_encoder(feats, prompts)["pred_tokens"]
            N = self.grid_size * self.grid_size
            D = pred_tokens.shape[-1]
            pred_tokens_bt = pred_tokens.view(batch_size, T * N, D)

            agg_tokens_bt = pred_tokens_bt.clone()

            for b in range(batch_size):
                groups = group_predictions(pred_tokens_bt[b], similarity_threshold=self.merge_similarity,
                                        min_component_size=1, merge_small_groups=False)
                for g in groups:
                    idx = torch.tensor(g, device=agg_tokens_bt.device, dtype=torch.long)
                    region_tok = pred_tokens_bt[b, idx].mean(dim=0)                       
                    agg_tokens_bt[b, idx] = region_tok   

            agg_tokens = agg_tokens_bt.view(batch_size, T, N, D).reshape(batch_size * T, N, D)
            tokens_grid = (agg_tokens.view(batch_size*T, self.grid_size, self.grid_size, D).permute(0, 3, 1, 2))
            logits_grid = self.decoder(tokens_grid)
            logits_up = F.interpolate(logits_grid, size=(H, W), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits_up, masks_flat.long(), ignore_index=255)
           
        return {
            'logits': logits_up,
            'loss': loss,
        }
    
    def validate(self):
        loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.step(batch)
                loss += outputs['loss']
        loss /= len(self.val_loader)
        return {'loss': loss}

    def train(self):
        iter_count = self.start_iter
        self.optimizer.zero_grad()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.decoder.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            running_loss = 0.0
            count = 0
            for batch in pbar:
                # Forward pass
                train_outputs = self.step(batch)
                train_loss = train_outputs['loss']

                # Backward pass
                self.scaler.scale(train_loss).backward()
                if self.max_grad_norm != -1:
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.max_grad_norm)
                if (iter_count + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                running_loss += train_loss.item()
                count += 1
                pbar.set_postfix(avg_loss=f"{running_loss / count:.4f}")

                # Log progress
                if (iter_count + 1) % self.logging_steps == 0:
                    val_outputs = self.validate()
                    val_loss = val_outputs['loss']
                    self.save_decoder(epoch, iter_count, val_loss)
                    if use_wandb:
                        wandb.log({
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                        })
                iter_count += 1


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()
