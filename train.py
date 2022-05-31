import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms as transforms

from utils.data_loading import OCTADataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

import pandas as pd

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def loss_fn(pred, true, sep=False):
    pred_td = torch.cat([pred[:, 0, 0, 4:6], pred[:, 0, 1, 2:8], pred[:, 0, 2, 1:9],
                         pred[:, 0, 3, 1:9], pred[:, 0, 4, :], pred[:, 0, 5, :],
                         pred[:, 0, 6, 1:9], pred[:, 0, 7, 1:9], pred[:, 0, 8, 2:8],
                         pred[:, 0, 9, 4:6]], 1)
    pred_pd = torch.cat([pred[:, 1, 0, 4:6], pred[:, 1, 1, 2:8], pred[:, 1, 2, 1:9],
                         pred[:, 1, 3, 1:9], pred[:, 1, 4, :], pred[:, 1, 5, :],
                         pred[:, 1, 6, 1:9], pred[:, 1, 7, 1:9], pred[:, 1, 8, 2:8],
                         pred[:, 1, 9, 4:6]], 1)

    pred_md = torch.mean(pred_td, 1)
    pred_psd = torch.mean(pred_pd, 1)

    true_td = true[:, 0:68]
    true_pd = true[:, 68:136]
    true_md = true[:, 136]
    true_psd = true[:, 137]

    l1_loss = nn.L1Loss()

    if sep:
        return l1_loss(pred_td, true_td), l1_loss(pred_pd, true_pd), \
               l1_loss(pred_md, true_md), l1_loss(pred_psd, true_psd)

    return l1_loss(pred_td, true_td) + l1_loss(pred_pd, true_pd) + \
           l1_loss(pred_md, true_md) + l1_loss(pred_psd, true_psd)


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 64,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset

    train_df = pd.read_csv('/data/updated_csv_0331_80_10_10/train_non_myopic.csv')
    val_df = pd.read_csv('/data/updated_csv_0331_80_10_10/val_non_myopic.csv')

    mean = [0.21159853, 0.21159853, 0.21159853]
    std = [0.00929382, 0.00929525, 0.00929995]

    train_transform = transforms.Compose([
        #     transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomRotation(5),
        transforms.Resize(200),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform = transforms.Compose([
        #     transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(200),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img_dir = "Dir_orig"
    if args.flip:
        img_dir = "Dir_flip"

    train_data = OCTADataset(train_df, img_dir, transform=train_transform)
    val_data = OCTADataset(val_df, img_dir, transform=transform)

    # 2. Split into train / validation partitions
    n_val = len(val_data)
    n_train = len(train_data)

    # 3. Create data loaders
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = loss_fn
    global_step = 0
    val_step = 0
    best_val_loss = 1e12

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0

        train_td_loss = 0.0
        train_pd_loss = 0.0
        train_md_loss = 0.0
        train_psd_loss = 0.0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for batch in train_dl:
                images = batch[0]
                true_masks = batch[1]

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    td_loss, pd_loss, md_loss, psd_loss = criterion(masks_pred, true_masks, sep=True)
                    loss = (1 - args.mean_wt) * (td_loss + pd_loss) + args.mean_wt * (md_loss + psd_loss)
                    train_td_loss += images.shape[0] * td_loss.item()
                    train_pd_loss += images.shape[0] * pd_loss.item()
                    train_md_loss += images.shape[0] * md_loss.item()
                    train_psd_loss += images.shape[0] * psd_loss.item()

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'comp_train_loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            cum_td_loss = 0.0
            cum_pd_loss = 0.0
            cum_md_loss = 0.0
            cum_psd_loss = 0.0
            cum_comp_loss = 0.0

            for batch in val_dl:
                images = batch[0]
                true_masks = batch[1]

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.no_grad():
                    masks_pred = net(images)
                    td_loss, pd_loss, md_loss, psd_loss = criterion(masks_pred, true_masks, sep=True)
                    loss = (1 - args.mean_wt) * (td_loss + pd_loss) + args.mean_wt * (md_loss + psd_loss)
                    cum_td_loss += images.shape[0] * td_loss.item()
                    cum_pd_loss += images.shape[0] * pd_loss.item()
                    cum_md_loss += images.shape[0] * md_loss.item()
                    cum_psd_loss += images.shape[0] * psd_loss.item()
                    cum_comp_loss += images.shape[0] * loss.item()

                experiment.log({
                    'comp_val_loss': loss.item(),
                    'step': val_step,
                    'epoch': epoch
                })

                val_step += 1

        print("Epoch: {} \nTrain td_loss: {} pd loss: {} md-10 loss: {} psd-10 loss: {}\n Val td_loss: {} "
              "pd loss: {} md-10 loss: {} psd-10 loss: {}\n".format(epoch,
                                                                    train_td_loss / n_train, train_pd_loss / n_train,
                                                                    train_md_loss / n_train, train_psd_loss / n_train,
                                                                    cum_td_loss / n_val, cum_pd_loss / n_val,
                                                                    cum_md_loss / n_val, cum_psd_loss / n_val))
        experiment.log({
            'train_td_loss': train_td_loss / n_train,
            'train_pd_loss': train_pd_loss / n_train,
            'train_md_loss': train_md_loss / n_train,
            'train_psd_loss': train_psd_loss / n_train,
            'val_td_loss': cum_td_loss / n_val,
            'val_pd_loss': cum_pd_loss / n_val,
            'val_md_loss': cum_md_loss / n_val,
            'val_psd_loss': cum_psd_loss / n_val,
            'epoch': epoch
        })
        cum_comp_loss = cum_comp_loss / n_val

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        if cum_comp_loss < best_val_loss:
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_best.pth'))
            logging.info(f'Best checkpoint saved at epoch {epoch}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--mean_wt', '-mwt', type=float, default=0.1, help='Loss weight for md-10 and psd-10')
    parser.add_argument('--flip', '-fl', type=bool, default=True, help='Flipped images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
