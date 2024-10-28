import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import Lamb
from ignite.handlers import PiecewiseLinear
from ddpm.unet import UNet
from tqdm import tqdm
from einops import rearrange


class DDPM:
    def __init__(self, T=1000):
        self.T = T
        self.unet = UNet(T=T).cuda()
        self.ema_model = swa_utils.AveragedModel(
            self.unet, device="cuda", multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.999))
        lr = 1e-3
        self.optimizer = Lamb(self.unet.parameters(), lr=lr)
        self.scheduler = PiecewiseLinear(
            self.optimizer, 'lr', [(4000, lr), (20000, 0.5*lr), (40000, 0.1*lr)])

        self._alpha = torch.sqrt(1 - 0.02 * torch.arange(1, T + 1) / T).cuda()
        self._beta = torch.sqrt(1 - self._alpha ** 2).cuda()
        self._bar_alpha = torch.cumprod(self._alpha, dim=0).cuda()
        self._bar_beta = torch.sqrt(1 - self._bar_alpha ** 2).cuda()
        self._sigma = self._beta.clone()

    def _l2_loss(self, x, y):
        return torch.sum((x - y) ** 2)

    def _make_train_pair(self, x):
        batch_size = x.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x.device)
        bar_alpha = self._bar_alpha[t][:, None, None, None]
        bar_beta = self._bar_beta[t][:, None, None, None]
        noise = torch.randn_like(x)
        noisy_x = bar_alpha * x + bar_beta * noise
        return noisy_x, t, noise

    def _save_checkpoint(self, outdir, epoch):
        print(f"Saving checkpoint at epoch {epoch}")
        os.makedirs(os.path.join(outdir, 'state'), exist_ok=True)
        torch.save(self.unet.state_dict(), os.path.join(
            outdir, f'state/unet_{epoch}.pt'))
        torch.save(self.ema_model.state_dict(),
                   os.path.join(outdir, f'state/ema_{epoch}.pt'))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(outdir, f'optimizer.pt'))
        torch.save(self.scheduler.state_dict(),
                   os.path.join(outdir, f'scheduler.pt'))
        
            
    def _load_checkpoint(self, outdir):
        # Find the latest checkpoint
        os.makedirs(os.path.join(outdir, 'state'), exist_ok=True)
        checkpoints = os.listdir(os.path.join(outdir, 'state'))
        checkpoints = [int(c.split('_')[1].split('.')[0]) for c in checkpoints]
        checkpoints.sort()
        if len(checkpoints) == 0:
            print("No checkpoint found")
            return 0
        latest_checkpoint = checkpoints[-1]
        
        # Load the latest checkpoint
        print(f"Loading checkpoint at epoch {latest_checkpoint}")
        if os.path.exists(os.path.join(outdir, f'state/unet_{latest_checkpoint}.pt')):
            self.unet.load_state_dict(torch.load(os.path.join(outdir, f'state/unet_{latest_checkpoint}.pt')))
        if os.path.exists(os.path.join(outdir, f'state/ema_{latest_checkpoint}.pt')):
            self.ema_model.load_state_dict(torch.load(os.path.join(outdir, f'state/ema_{latest_checkpoint}.pt')))
        if os.path.exists(os.path.join(outdir, f'optimizer.pt')):
            self.optimizer.load_state_dict(torch.load(os.path.join(outdir, f'optimizer.pt')))
        if os.path.exists(os.path.join(outdir, f'scheduler.pt')):
            self.scheduler.load_state_dict(torch.load(os.path.join(outdir, f'scheduler.pt')))
            
        return latest_checkpoint + 1

    def train(self, dataloader, outdir, epochs=100, steps_per_epoch=2000, checkpoint_freq=1, resume=False, start_epoch=0):
        os.makedirs(outdir, exist_ok=True)
        writer = SummaryWriter(outdir)
        if resume:
            start_epoch = self._load_checkpoint(outdir)

        for epoch in tqdm(range(start_epoch, epochs), desc='Training'):
            pbar = tqdm(range(steps_per_epoch),
                        desc=f'Epoch {epoch}')
            epoch_loss = 0
            epoch_samples = 0
            dataloader_it = iter(dataloader)
            for step in pbar:
                batch = next(dataloader_it, None)
                if batch is None:
                    dataloader_it = iter(dataloader)
                    batch = next(dataloader_it)
                x = batch[0].cuda()

                noisy_x, t, noise = self._make_train_pair(x)
                self.optimizer.zero_grad()
                noise_pred = self.unet(noisy_x, t)
                loss = self._l2_loss(noise_pred, noise)
                loss.backward()
                self.optimizer.step()
                self.scheduler(None)
                self.ema_model.update_parameters(self.unet)

                epoch_loss += loss.item()
                epoch_samples += x.shape[0]
                current_loss = loss.item() / x.shape[0]
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})
                writer.add_scalar('step_loss', current_loss, epoch * steps_per_epoch + step)

            epoch_loss /= epoch_samples
            print(f"Epoch {epoch} loss: {epoch_loss:.4f}")
            writer.add_scalar('epoch_loss', epoch_loss, epoch)

            if epoch % checkpoint_freq == 0:
                self._save_checkpoint(outdir, epoch)
                
            samples_ema = self.ddpm_sample(16, t0=0, use_ema=True)
            writer.add_images('samples_ema', self.make_samples_grid(samples_ema), epoch)
            samples_no_ema = self.ddpm_sample(16, t0=0, use_ema=False)
            writer.add_images('samples_no_ema', self.make_samples_grid(samples_no_ema), epoch)
                
        writer.close()
        
    def ddpm_sample(self, n, z=None, t0=0, use_ema=True):
        self.ema_model.eval()
        with torch.no_grad():
            if z is None:
                z = torch.randn(n, 3, 128, 128, device='cuda')
            else:
                z = z.clone().detach().cuda()
            
            for t in tqdm(range(t0, self.T), desc='Sampling'):
                t = self.T - t - 1
                t = torch.full((n,), t, device='cuda')
                noise_pred = self.ema_model(z, t) if use_ema else self.unet(z, t)
                alpha = self._alpha[t][:, None, None, None]
                beta = self._beta[t][:, None, None, None]
                bar_beta = self._bar_beta[t][:, None, None, None]
                sigma = self._sigma[t][:, None, None, None]
                
                # DDPM sampling
                z -= beta**2 / bar_beta * noise_pred
                z /= alpha
                z += sigma * torch.randn_like(z)
                
            x = z.clamp(-1, 1)
        self.ema_model.train()
        return x
                    
    def make_samples_grid(self, samples, grid_size=4):
        figure = rearrange(samples, '(gh gw) c h w -> 1 c (gh h) (gw w)', gh=grid_size, gw=grid_size)
        figure = (figure + 1) / 2
        return figure.cpu()