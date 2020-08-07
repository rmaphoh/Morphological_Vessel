import torch.nn.functional as F
from dice_loss import dice_coeff
import argparse
import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from model import UNet, Generator
from dataset import BasicDataset, val_transformer
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from PIL import Image
from scipy.special import expit


def test_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    #net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    num = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            print('The type of img,', imgs.dtype)
            with torch.no_grad():
                mask_pred = net(imgs)
                #mask_pred = mask_pred*255

                mask_pred_tensor = mask_pred.clone().detach()
                mask_pred_tensor = torch.sigmoid(mask_pred_tensor)
                mask_pred_tensor = torch.squeeze(mask_pred_tensor)
                save_image(mask_pred_tensor, './seg_results/Unet_GAN/img_{:02}.png'.format(num))

                mask_pred_numpy = mask_pred.clone().detach().cpu().numpy()
                print(np.shape(mask_pred_numpy))
                print(np.unique(mask_pred_numpy))
                #mask_pred_numpy = np.squeeze(mask_pred_numpy, axis=0)
                mask_pred_numpy = expit(mask_pred_numpy)
                mask_pred_numpy = np.squeeze(mask_pred_numpy)
                print(np.shape(mask_pred_numpy))
                print(np.unique(mask_pred_numpy))
                k = (mask_pred_numpy*255).astype(np.uint8)
                #k = k.transpose((1, 2, 0))
                print(np.shape(k))
                result = Image.fromarray(k)
                result.save('./seg_results/Unet_GAN/PIL_{:02}.png'.format(num))

                



            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                num += 1
                single_loss = dice_coeff(pred, true_masks).item()
                print('This is the image: ', num)
                print('The F1 is: ',single_loss )
                tot += single_loss
            pbar.update()
    print('#################', tot / n_val)
    net.train()
    return 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
dis = 'unet'
dataset = 'DRIVE_AV'
gan2seg = 13
image_size = (592,592)
batch_size = 1
test_dir= "./data/{}/test/images/".format(dataset)
test_mask = "./data/{}/test/2nd_manual/".format(dataset)
img_out_dir="./{}/AV_segmentation_results_{}_{}".format(dataset,dis,gan2seg)

dataset = BasicDataset(test_dir, test_mask, image_size, val_transformer)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

#net = UNet(n_channels=3, n_classes=1, bilinear=True)
net = Generator(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)

net.load_state_dict(torch.load('./DRIVE_AV/checkpoints/AV_model_unet_13CP_epoch21.pth'))
net.eval()
net.to(device=device)
test_net(net=net, loader=test_loader, device=device)

