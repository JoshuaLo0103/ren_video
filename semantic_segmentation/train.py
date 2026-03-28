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
from dataloader import VOCSegmentation, ADESegmentation
from models import VOCDecoderLinear, ADEDecoderLinear

sys.path.append('..')
sys.path.append('../segment_anything/')
from model import FeatureExtractor, RegionEncoder


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
        if config['data']['target_data'] == 'pascal_voc':
            dataset = VOCSegmentation(config, image_set='trainval')
            train_size = int(0.95 * len(dataset))
            val_size = len(dataset) - train_size
            train_subset, val_subset = random_split(dataset, [train_size, val_size])
            self.train_loader = DataLoader(train_subset, batch_size=config['parameters']['batch_size'],
                                           num_workers=config['parameters']['num_workers'], shuffle=True, pin_memory=True)
            self.val_loader = DataLoader(val_subset, batch_size=config['parameters']['batch_size'],
                                         num_workers=config['parameters']['num_workers'], pin_memory=True)
        elif config['data']['target_data'] == 'ade20k':
            train_dataset = ADESegmentation(config, split='training')
            val_dataset = ADESegmentation(config, split='validation')
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

        # Create the models
        self.extractor_name = config['ren']['pretrained']['feature_extractors'][0]
        self.patch_size = config['ren']['pretrained']['patch_sizes'][0]
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device)
        if config['data']['target_data'] == 'pascal_voc':
            self.decoder = VOCDecoderLinear(config).to(device)
        elif config['data']['target_data'] == 'ade20k':
            self.decoder = ADEDecoderLinear(config).to(device)

        # Create prompts for region encoder
        self.image_resolution = config['ren']['parameters']['image_resolution']
        self.grid_size = self.image_resolution // self.patch_size
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
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
        batch_size = images.shape[0]

        with autocast(dtype=torch.bfloat16):
            # Compute outputs
            with torch.no_grad():
                _, feature_maps = self.feature_extractor(self.extractor_name, images, resize=False)
                prompts = [self.grid_points for _ in range(batch_size)]
                region_tokens = self.region_encoder(feature_maps, prompts)['pred_tokens']
                region_tokens = region_tokens.view(batch_size, self.grid_size, self.grid_size, -1)
            outputs = self.decoder(region_tokens.permute(0, 3, 1, 2))

            # Resize the outputs to the desired dimensions
            resized_outputs = torch.nn.functional.interpolate(outputs, size=[self.image_resolution, self.image_resolution],
                                                              mode='bilinear')
            resized_outputs = resized_outputs.flatten(-2).permute(0, 2, 1).reshape(-1, outputs.shape[1])
            
            # Compute loss
            targets = masks.view(-1)
            loss = F.cross_entropy(resized_outputs, targets, ignore_index=255, reduction='mean')
            
        return {
            'outputs': outputs,
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
            for batch in tqdm(self.train_loader, desc=f'Running epoch {epoch}'):
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