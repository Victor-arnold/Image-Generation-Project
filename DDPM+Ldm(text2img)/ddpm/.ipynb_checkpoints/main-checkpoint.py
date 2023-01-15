# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 22:22:28 2022

@author: nkliu
"""

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from utils import *

if __name__ == '__main__':
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).cuda()
    
    trainer = Trainer(
        diffusion,
        'cifar10_horse/train/',
        image_size = 32,
        train_batch_size = 4,
        train_lr = 2e-5,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )
    
    trainer.train()
    

