import argparse
import logging
import sys
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from utils.data_loading import OCTADataset
from unet import UNet
from train import loss_fn
import unet

import pandas as pd

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def test_net(net,
             device,
             batch_size: int = 64,
             val_percent: float = 0.1,
             save_checkpoint: bool = True,
             img_scale: float = 0.5,
             amp: bool = False):
    # 1. Create dataset

    test_df = pd.read_csv('/data/updated_csv_0331_80_10_10/test_non_myopic.csv')

    mean = [0.21159853, 0.21159853, 0.21159853]
    std = [0.00929382, 0.00929525, 0.00929995]

    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(200),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img_dir = "Dir_orig"
    if args.flip:
        img_dir = "Dir_flip"

    test_data = OCTADataset(test_df, img_dir, transform=transform)

    # 2. Split into train / validation partitions
    n_test = len(test_data)

    # 3. Create data loaders
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(batch_size=batch_size, val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  img_scale=img_scale, amp=amp))

    logging.info(f'''Starting training:
        Batch size:      {batch_size}
        Test size:       {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    criterion = loss_fn
    val_step = 0

    cum_td_loss = 0.0
    cum_pd_loss = 0.0
    cum_md_loss = 0.0
    cum_psd_loss = 0.0

    for batch in test_dl:
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

        experiment.log({
            'comp_val_loss': loss.item(),
            'step': val_step,
        })

        val_step += 1

    print("Test td_loss: {} pd loss: {} md-10 loss: {} psd-10 loss: {}".format(cum_td_loss / n_test, cum_pd_loss / n_test,
                                                                             cum_md_loss / n_test, cum_psd_loss / n_test))

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
    net.outc = unet.unet_parts.OutConvOurs(64, args.classes, 200)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        test_net(net=net,
                 batch_size=args.batch_size,
                 device=device,
                 img_scale=args.scale,
                 val_percent=args.val / 100,
                 amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
