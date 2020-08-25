import torch.nn.functional as F
from dice_loss import dice_coeff
import argparse
import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from model import UNet, Generator_main
#from dataset import BasicDataset, val_transformer
from server_code.dataset import BasicDataset,val_transformer
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from PIL import Image
from scipy.special import expit
from eval import eval_net
from skimage import filters


def test_net(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    #net.eval()
    #mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32 if net.n_classes == 1 else torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    num = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
        
            with torch.no_grad():
                mask_pred,_,_,_ = net(imgs)
                #mask_pred = mask_pred*255

                mask_pred_tensor = mask_pred.clone().detach()
                mask_pred_tensor = torch.sigmoid(mask_pred_tensor)
                mask_pred_tensor = torch.squeeze(mask_pred_tensor)
                save_image(mask_pred_tensor, './seg_results/20200817_addgan_av/img_{:02}.png'.format(num))

                mask_pred_numpy = mask_pred.clone().detach().cpu().numpy()
                #print(np.shape(mask_pred_numpy))
                #print(np.unique(mask_pred_numpy))
                #mask_pred_numpy = np.squeeze(mask_pred_numpy, axis=0)
                mask_pred_numpy = expit(mask_pred_numpy)
                mask_pred_numpy = np.squeeze(mask_pred_numpy)
                threshold=filters.threshold_otsu(mask_pred_numpy)
                mask_pred_numpy_bin=np.zeros(mask_pred_numpy.shape)
                mask_pred_numpy_bin[mask_pred_numpy>=threshold]=1
                mask_pred_numpy = mask_pred_numpy_bin

                #print(np.shape(mask_pred_numpy))
                #print(np.unique(mask_pred_numpy))
                k = (mask_pred_numpy*255).astype(np.uint8)
                k = k.transpose((1, 2, 0))
                #print(np.shape(k))
                result = Image.fromarray(k)
                result.save('./seg_results/20200817_addgan_av/PIL_{:02}.png'.format(num))
                
                pbar.update()

            num +=1

    epoch = 0


    if mode!='vessel':

        val_score, acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr = eval_net(epoch, net, loader=test_loader, device=device, mask=True, mode = mode)
    else:

        val_score, acc, sensitivity, specificity, precision, G, F1_score_2 = eval_net(epoch, net, loader=test_loader, device=device, mask=True, mode='vessel')
    print('Accuracy: ', acc)
    print('Sensitivity: ', sensitivity)
    print('specificity: ', specificity)
    print('precision: ', precision)
    print('G: ', G)
    print('F1_score_2: ', F1_score_2)
    if mode != 'vessel':
        print('auc_roc: ', auc_roc)
        print('auc_pr: ', auc_pr)
    
    print('################################')
    #net.train()
    if mode != 'vessel':
        return acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr
    else:
        return acc, sensitivity, specificity, precision, G, F1_score_2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
dis = 'unet'
dataset = 'DRIVE_AV'
gan2seg = 13
image_size = (592,592)
batch_size = 1
test_dir= "./data/{}/test/images/".format(dataset)
test_mask = "./data/{}/test/1st_manual/".format(dataset)
#test_mask = "./data/{}/test/2nd_manual/".format(dataset)
img_out_dir="./{}/AV_segmentation_results_{}_{}".format(dataset,dis,gan2seg)
mode = 'vessel'

acc_total = []
sensitivity_total = []
specificity_total = []
precision_total = []
G_total = []
F1_score_2_total = []
auc_roc_total = []
auc_pr_total = []

dataset = BasicDataset(test_dir, test_mask, image_size, val_transformer, train_or=False)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

#net = UNet(n_channels=3, n_classes=1, bilinear=True)
net = Generator_main(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)

#net = Generator(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)

########################3
#########################3
##########################       20200812_unetgan    ###########################
for i in range(10):
    net.load_state_dict(torch.load('./DRIVE_AV/checkpoints/20200824_all/AV_model_unet_13CP_epoch{}.pth'.format(511+10*i)))
    net.eval()
    net.to(device=device)
    if mode != 'vessel':
        acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr = test_net(net=net, loader=test_loader, device=device, mode=mode)
    else:
        acc, sensitivity, specificity, precision, G, F1_score_2 = test_net(net=net, loader=test_loader, device=device, mode=mode)
    
    '''
    acc_total += acc
    sensitivity_total += sensitivity
    specificity_total += specificity
    precision_total += precision
    G_total += G
    F1_score_2_total += F1_score_2
    if mode != 'vessel':
        auc_roc_total += auc_roc
        auc_pr_total += auc_pr
    '''

    acc_total.append(acc)
    sensitivity_total.append(sensitivity)
    specificity_total.append(specificity)
    precision_total.append(precision)
    G_total.append(G)
    F1_score_2_total.append(F1_score_2)
    if mode != 'vessel':
        auc_roc_total.append(auc_roc)
        auc_pr_total.append(auc_pr)
#############################################3
print('Accuracy: ', np.mean(acc_total))
print('Sensitivity: ', np.mean(sensitivity_total))
print('specificity: ', np.mean(specificity_total))
print('precision: ', np.mean(precision_total))
print('G: ', np.mean(G_total))
print('F1_score_2: ', np.mean(F1_score_2_total))

print('Accuracy: ', np.std(acc_total))
print('Sensitivity: ', np.std(sensitivity_total))
print('specificity: ', np.std(specificity_total))
print('precision: ', np.std(precision_total))
print('G: ', np.std(G_total))
print('F1_score_2: ', np.std(F1_score_2_total))


if mode != 'vessel':
    
    print('auc_roc: ', np.mean(auc_roc_total))
    print('auc_pr: ', np.mean(auc_pr_total))
    print('auc_roc: ', np.std(auc_roc_total))
    print('auc_pr: ', np.std(auc_pr_total))
