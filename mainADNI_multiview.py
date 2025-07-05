import logging
import sys
import json
from argparse import ArgumentParser
from dataADNI_multiview import ADNIDataset

#from modeling_m3d_clip import ViT
from modelADNI import ADNIModel

import torchio as tio
import os
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

import numpy as np
import torch
import random

import math

import time

from torch.utils.tensorboard import SummaryWriter

# for HPT
from hpt.models.policy import Policy
from hydra import compose, initialize


def parse_args(args=None):
    parser = ArgumentParser()

    ## Required parameters for data module
    parser.add_argument("--data_dir", default="jsons/", type=str)
    parser.add_argument("--split_json_train", default="dataset.json", type=str)
    parser.add_argument("--split_json_eval", default="dataset.json", type=str)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    #parser.add_argument("--clip_range", default=(-175, 250), type=int, nargs="+")
    #parser.add_argument("--mean_std", default=None, type=float, nargs="+")

    ## Required parameters for model module
    parser.add_argument("--img_size", default=(256, 256, 32), type=int, nargs="+")
    
    #parser.add_argument("--dropout_rate", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--warmup_steps", default=20, type=int)
    parser.add_argument("--max_steps", default=25000, type=int)

    ## Required parameters for trainer module
    parser.add_argument("--default_root_dir", default=".", type=str)
    parser.add_argument("--gpus", default=-1, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--check_val_every_n_epoch", default=100, type=int)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--log_every_n_steps", default=1, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--accelerator", default="ddp", type=str)
    parser.add_argument("--seed", default=1234, type=int)


    ## Require parameters for evaluation
    parser.add_argument("--evaluation", default=0, type=int)
    parser.add_argument("--data_augmentation", default=1, type=int)
    parser.add_argument("--down_resolution", default=0, type=int)

    ## stk:
    ## for training:
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--lr_drop", default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--inbalanced_sampler", default=0, type=int)
    parser.add_argument("--inbalanced_sampler_weight_path", default=None, type=str)
    
    ## resume training:
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to load the resume training from.')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Epoch to start the training from.')
    
    ## save model:
    parser.add_argument('--to_save_path', default=None, type=str,
                        help='Path to save the trained model.')

    
    ## multi-view data:
    parser.add_argument("--volume", default=1, type=int)
    parser.add_argument("--parcellation", default=0, type=int)
    
    ## HPT model and our model:
    parser.add_argument("--trunk_pretrained_path", default=None, type=str)
    parser.add_argument("--image_encoder_pretrained_path", default=None, type=str)
    parser.add_argument("--share_image_encoder", default=1, type=int)
    parser.add_argument("--use_modality_tokens", default=0, type=int)

    ## freeze trunk:
    parser.add_argument("--freeze_trunk", default=0, type=int)
    
    ## new training pipeline:
    parser.add_argument("--change_bias", default=0, type=int)
    
    ## turn off BN:
    parser.add_argument("--batch_normalization", default=1, help="1 means turn on BN. Default is 1.", type=int)
    
    args = parser.parse_args(args)
    return args


def main(args):
    """
    wandb_logger = pl.loggers.WandbLogger(
        project="MedicalSegmentation", config=vars(args), log_model=False
    )

    pl.seed_everything(args.seed)
    """
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # augmentation:
    """
    # stk: we will currently not use randomelastic and we need to fix the rotation and translation axis for random affine
    spatial_transforms = {
        tio.RandomElasticDeformation(): 0.2,
        tio.RandomAffine(): 0.8,
    }
    """
    """
    spatial_transforms = {
        tio.RandomAffine(scales=(0.1, 0.1, 0), degrees=(0, 0, 10), translation=(20, 20, 0)): 1,
    }
    """
    
    """
    spatial_transforms = {
        tio.RandomAffine(scales=(0.1, 0, 0.1), degrees=(0, 10, 0), translation=(20, 0, 20)): 1.
    } # left - right, back - front, bottom - top
    """
    # now we take full 3D augmentation and then crop and pad
    spatial_transforms = {
        tio.RandomAffine(scales=(0.1, 0.1, 0.1), degrees=(10, 10, 10), translation=(20, 20, 20)): 1.
    } # left - right, back - front, bottom - top
    
    

    """
    transform = tio.Compose([
        #tio.OneOf(spatial_transforms,  p=0.5 if not args.evaluation else 0),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.Resize([args.img_size, args.img_size, 5] if type(args.img_size) is int 
                          else list(args.img_size)) # tio requires: W, H, D 
    ])
    """
    transform = tio.Compose([
                    tio.ToCanonical(),
                    tio.OneOf(spatial_transforms, p=0.5 if (not args.evaluation) and args.data_augmentation else 0), # for augmentation
                    tio.RescaleIntensity(out_min_max=(0, 1)),
                    #tio.RandomNoise(p=0.5 if (not args.evaluation) and args.data_augmentation else 0), # for augmentation
                    tio.Resample((2, 2, 2), p=1 if args.down_resolution else 0),
    ])
    
    transform_valid = tio.Compose([
                    tio.ToCanonical(),
                    tio.RescaleIntensity(out_min_max=(0, 1)),
                    tio.Resample((2, 2, 2), p=1 if args.down_resolution else 0),
    ])
    

    if args.inbalanced_sampler:
        sample_weight = np.load(os.path.join(args.data_dir, args.inbalanced_sampler_weight_path))
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
        
    train_dataset = ADNIDataset(args.data_dir, args.split_json_train, parcellation=args.parcellation, volume=args.volume, transform=transform, img_size=args.img_size)
    eval_dataset = ADNIDataset(args.data_dir, args.split_json_eval, parcellation=args.parcellation, volume=args.volume, transform=transform_valid, img_size=args.img_size)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.train_batch_size,
                              sampler=None if not args.inbalanced_sampler else sampler,
                              shuffle=True if not args.inbalanced_sampler else False) # stk: shuffle is mutually exclusive with sampler
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    
    ########################################################################
    # build from HPT
    policy = Policy.from_pretrained("hf://liruiw/hpt-base")

    domain = "mujoco_metaworld"
    with initialize(version_base="1.2", config_path="./hpt_pretrained_model"):
        cfg = compose(config_name="config_modify", overrides=[])
        
    policy.init_domain_stem(domain, cfg.stem)
    
    policy.finalize_modules()
    print("HPT model:", flush=True)
    policy.print_model_stats()
    
    if args.trunk_pretrained_path is not None:
        # this is for the original HPT model trunk part
        checkpoint = torch.load(args.trunk_pretrained_path, map_location='cpu')
        message=policy.trunk.load_state_dict(checkpoint, strict=True)

        print("Load trunk ckpt:", message)
    ########################################################################
    # build our model based on HPT:
    model = ADNIModel(trunk = policy.trunk["trunk"],
                      image_stem = policy.stems['mujoco_metaworld_image'],
                      image_encoder_depth=18,
                      image_encoder_pretrained_path=args.image_encoder_pretrained_path,
                      share_image_encoder=args.share_image_encoder,
                      state_input_dim=280,
                      modality_embed_dim=256,
                      modality_names_types={"sag": "image",
                                          "cor": "image",
                                          "axi": "image",
                                          "volume": "state"
                                         },
                      use_modality_tokens=args.use_modality_tokens,
                      batch_normalization=args.batch_normalization,
                     )
    
    ########################################################################
    # give a better initialization of the bias in model:
    if args.change_bias:
        model.head.fc3.bias = torch.nn.parameter.Parameter(data=torch.tensor([68.5], device=device), requires_grad=True) # 68.5 is the mean from the training distribution
    
    ########################################################################
    
    model.to(device)
    """
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    """
 
    if args.freeze_trunk:
        for n, p in model.trunk.named_parameters():
            p.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    criterion = torch.nn.MSELoss()

    # resume model from pretrain:
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        message = model.load_state_dict(checkpoint['model'], strict=True)
        
        print(message)
        
        if not args.evaluation and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.evaluation:
        evaluate(model, criterion, eval_loader, device)
        exit()
    
    print("Start training")
    writer = SummaryWriter(log_dir=args.to_save_path)
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        """
        if args.distributed:
            sampler_train.set_epoch(epoch)
        """
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            args.clip_max_norm, writer)
        lr_scheduler.step()
        if args.to_save_path is not None:
            checkpoint_paths = [os.path.join(args.to_save_path, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(os.path.join(args.to_save_path, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    #'model': model_without_ddp.state_dict(),
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        evaluate(
            model, criterion, eval_loader, device, epoch, writer
        )
    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, criterion,
                    data_loader, optimizer,
                    device, epoch, max_norm, writer=None):
    model.train()
    criterion.train()
    print("epoch:", epoch, flush=True)
    print_freq = 50
    total_loss = 0
    total_loss_log = 0
    for itr, (sag, cor, axi, vol, labels) in enumerate(data_loader):
        sag = sag.to(device).float()
        cor = cor.to(device).float()
        axi = axi.to(device).float()
        vol = vol.to(device).float()
        labels = labels.to(device).float().unsqueeze(-1)
        
        data = {'sag': sag.repeat([1, 3, 1, 1, 1]).permute([0, 1, -1, -3, -2]),
                'cor': cor.repeat([1, 3, 1, 1, 1]).permute([0, 1, -1, -3, -2]),
                'axi': axi.repeat([1, 3, 1, 1, 1]).permute([0, 1, -1, -3, -2]),
                'volume': vol,
                }
        
        
        #with torch.cuda.amp.autocast():
        outputs = model(data)
        losses = criterion(outputs, labels)

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("prediction is:", outputs)
            # save input:
            torch.save({"input": data,
                        "model": model.state_dict()}, "./infinite.pth")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        
        total_loss += losses
        
        if itr != 0 and itr % print_freq == 0:
            print("training loss: {:0.2f}".format((total_loss / print_freq).item()), flush=True)
            if writer is not None:
                writer.add_scalar('training loss',
                              (total_loss / print_freq).item(),
                              epoch * len(data_loader) + itr)
            total_loss_log += total_loss / print_freq
            total_loss = 0
            
    print("Average loss: {:0.2f}".format((total_loss_log / int(itr / print_freq)).item()), flush=True)
    
def evaluate(model, criterion, data_loader, device, epoch=0, writer=None):
    model.eval()
    criterion.eval()

    print_freq = 5
    total_MSE_loss = 0
    total_MAE_loss = 0
    total_MSE_loss_log = 0
    total_MAE_loss_log = 0
    
    validation_criterion = torch.nn.L1Loss()
    
    with torch.no_grad():
         for itr, (sag, cor, axi, vol, labels) in enumerate(data_loader):
            sag = sag.to(device).float()
            cor = cor.to(device).float()
            axi = axi.to(device).float()
            vol = vol.to(device).float()
            labels = labels.to(device).float().unsqueeze(-1)

            data = {'sag': sag.repeat([1, 3, 1, 1, 1]).permute([0, 1, -1, -3, -2]),
                    'cor': cor.repeat([1, 3, 1, 1, 1]).permute([0, 1, -1, -3, -2]),
                    'axi': axi.repeat([1, 3, 1, 1, 1]).permute([0, 1, -1, -3, -2]),
                    'volume': vol,
                    }


            #with torch.cuda.amp.autocast():
            outputs = model(data)

            MSE_losses = criterion(outputs, labels)

            MAE_losses = validation_criterion(outputs, labels)

            total_MSE_loss += MSE_losses
            total_MAE_loss += MAE_losses

            if itr != 0 and itr % print_freq == 0:
                print("validation MSE: {:0.2f}".format((total_MSE_loss / print_freq).item()), flush=True)  # mean squared error
                print("validation MAE: {:0.2f}".format((total_MAE_loss / print_freq).item()), flush=True)  # mean absolute error

                total_MSE_loss_log += total_MSE_loss / print_freq
                total_MAE_loss_log += total_MAE_loss / print_freq
                total_MSE_loss = 0
                total_MAE_loss = 0

    print("Average MSE loss: {:0.2f}".format((total_MSE_loss_log / int(itr / print_freq)).item()), flush=True)
    print("Average MAE loss: {:0.2f}".format((total_MAE_loss_log / int(itr / print_freq)).item()), flush=True)
    
    if writer is not None:
        writer.add_scalar('validation loss', (total_MSE_loss_log / int(itr / print_freq)).item(), epoch)

if __name__ == "__main__":
    args = parse_args()
    print(args, flush=True)
    main(args)