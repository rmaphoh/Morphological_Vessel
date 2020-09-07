import torch.nn.functional as F
import argparse
import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from model import UNet, Generator_main, Generator_branch
#from dataset import BasicDataset, val_transformer
from server_code.dataset import BasicDataset,val_transformer
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from PIL import Image
from scipy.special import expit
from eval import eval_net
from skimage import filters


def test_net(net_all, net_a, net_v, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    #net.eval()
    #mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32 if net_all.n_classes == 1 else torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    num = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
        
            with torch.no_grad():
                
                pre_a = net_a(imgs)
                pre_a_sigmoid= torch.sigmoid(pre_a)

                pre_v = net_v(imgs)
                pre_v_sigmoid= torch.sigmoid(pre_v)

                mask_pred, _, _, _ = net_all(imgs, pre_a_sigmoid, pre_v_sigmoid)
                #mask_pred,_,_,_ = net_all(imgs,)
                #mask_pred = mask_pred*255

                mask_pred_tensor = mask_pred.clone().detach()
                mask_pred_tensor = torch.sigmoid(mask_pred_tensor)
                mask_pred_tensor = torch.squeeze(mask_pred_tensor)
                save_image(mask_pred_tensor, './seg_results/20200907_all/img_{:02}.png'.format(num))

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
                result.save('./seg_results/20200907_all/PIL_{:02}.png'.format(num))
                
                pbar.update()

            num +=1

    epoch = 0


    if mode!='vessel':

        val_score, tot, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
            val_score_a , tot_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
                val_score_v , tot_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
                    val_score_u , tot_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u  = eval_net(epoch, net_all, net_a, net_v, loader=test_loader, device=device, mask=True, mode = mode, train_or='val')
    else:

        val_score, acc, sensitivity, specificity, precision, G, F1_score_2 = eval_net(epoch, net_all, net_a, net_v, loader=test_loader, device=device, mask=True, mode='vessel')
    print('Accuracy: ', tot)
    print('Sensitivity: ', sent)
    print('specificity: ', spet)
    print('precision: ', pret)
    print('G: ', G_t)
    print('F1_score_2: ', F1t)
    print('MSE: ', mset)
    print('IOU: ', iout)
    if mode != 'vessel':
        print('auc_roc: ', auc_roct)
        print('auc_pr: ', auc_prt)
    
    print('################################')
    #net.train()
    if mode != 'vessel':
        #return acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr, mse
        return val_score, tot, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
                val_score_a , tot_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
                val_score_v , tot_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
                val_score_u , tot_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u
    else:
        return acc, sensitivity, specificity, precision, G, F1_score_2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
dis = 'unet'
dataset = 'LES-AV'
gan2seg = 13
image_size = (592,592)
batch_size = 1
test_dir= "./data/{}/test/images/".format(dataset)
test_mask = "./data/{}/test/1st_manual/".format(dataset)
module_dir =  "./data/{}/test/mask/".format(dataset)
img_out_dir="./{}/AV_segmentation_results_{}_{}".format(dataset,dis,gan2seg)
mode = 'whole'

acc_total_a = []
sensitivity_total_a = []
specificity_total_a = []
precision_total_a = []
G_total_a = []
F1_score_2_total_a = []
auc_roc_total_a = []
auc_pr_total_a = []
mse_total_a = []
iou_total_a = []

acc_total_v = []
sensitivity_total_v = []
specificity_total_v = []
precision_total_v = []
G_total_v = []
F1_score_2_total_v = []
auc_roc_total_v = []
auc_pr_total_v = []
mse_total_v = []
iou_total_v = []

acc_total_u = []
sensitivity_total_u = []
specificity_total_u = []
precision_total_u = []
G_total_u = []
F1_score_2_total_u = []
auc_roc_total_u = []
auc_pr_total_u = []
mse_total_u = []
iou_total_u = []

acc_total = []
sensitivity_total = []
specificity_total = []
precision_total = []
G_total = []
F1_score_2_total = []
auc_roc_total = []
auc_pr_total = []
mse_total = []
iou_total = []

dataset = BasicDataset(test_dir, test_mask, module_dir, image_size, dataset_name=dataset, transforms=val_transformer, train_or=False)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

#net = UNet(n_channels=3, n_classes=1, bilinear=True)
net_G = Generator_main(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)


net_G_A = Generator_branch(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)
#net_D_A = Discriminator(input_channels=6, n_filters = 32, n_classes=3, bilinear=False)

net_G_V = Generator_branch(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)


#net = Generator(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)

########################3
#########################3
##########################       20200812_unetgan    ###########################
for i in range(10):
    net_G.load_state_dict(torch.load('./LES-AV/checkpoints/20200907_all/AV_model_unet_13CP_epoch{}_all.pth'.format(1301+10*i)))
    net_G_A.load_state_dict(torch.load('./LES-AV/checkpoints/20200907_all/AV_model_unet_13CP_epoch{}_A.pth'.format(1301+10*i)))
    net_G_V.load_state_dict(torch.load('./LES-AV/checkpoints/20200907_all/AV_model_unet_13CP_epoch{}_V.pth'.format(1301+10*i)))
    net_G.eval()
    net_G_A.eval()
    net_G_V.eval()
    net_G.to(device=device)
    net_G_A.to(device=device)
    net_G_V.to(device=device)

    if mode != 'vessel':
        #acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr, mse = test_net(net_all=net_G, net_a=net_G_A, net_v=net_G_V, loader=test_loader, device=device, mode=mode)
        val_score, acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
            val_score_a , acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
            val_score_v , acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
            val_score_u , acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u = test_net(net_all=net_G, net_a=net_G_A, net_v=net_G_V, loader=test_loader, device=device, mode=mode)
    else:
        acc, sensitivity, specificity, precision, G, F1_score_2 = test_net(net_all=net_G, net_a=net_G_A, net_v=net_G_V, loader=test_loader, device=device, mode=mode)
    
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
#########################################3
    acc_total_a.append(acc_a)
    sensitivity_total_a.append(sent_a)
    specificity_total_a.append(spet_a)
    precision_total_a.append(pret_a)
    G_total_a.append(G_t_a)
    F1_score_2_total_a.append(F1t_a)
    mse_total_a.append(mset_a)
    iou_total_a.append(iout_a)
    if mode != 'vessel':
        auc_roc_total_a.append(auc_roct_a)
        auc_pr_total_a.append(auc_prt_a)

###########################################
    acc_total_v.append(acc_v)
    sensitivity_total_v.append(sent_v)
    specificity_total_v.append(spet_v)
    precision_total_v.append(pret_v)
    G_total_v.append(G_t_v)
    F1_score_2_total_v.append(F1t_v)
    mse_total_v.append(mset_v)
    iou_total_v.append(iout_v)
    if mode != 'vessel':
        auc_roc_total_v.append(auc_roct_v)
        auc_pr_total_v.append(auc_prt_v)

############################################
    acc_total_u.append(acc_u)
    sensitivity_total_u.append(sent_u)
    specificity_total_u.append(spet_u)
    precision_total_u.append(pret_u)
    G_total_u.append(G_t_u)
    F1_score_2_total_u.append(F1t_u)
    mse_total_u.append(mset_u)
    iou_total_u.append(iout_u)
    if mode != 'vessel':
        auc_roc_total_u.append(auc_roct_u)
        auc_pr_total_u.append(auc_prt_u)

###########################################
    acc_total.append(acc)
    sensitivity_total.append(sent)
    specificity_total.append(spet)
    precision_total.append(pret)
    G_total.append(G_t)
    F1_score_2_total.append(F1t)
    mse_total.append(mset)
    iou_total.append(iout)
    if mode != 'vessel':
        auc_roc_total.append(auc_roct)
        auc_pr_total.append(auc_prt)

#############################################3

print('########################################3')
print('ARTERY')
print('#########################################')

print('Accuracy: ', np.mean(acc_total_a))
print('Sensitivity: ', np.mean(sensitivity_total_a))
print('specificity: ', np.mean(specificity_total_a))
print('precision: ', np.mean(precision_total_a))
print('G: ', np.mean(G_total_a))
print('F1_score_2: ', np.mean(F1_score_2_total_a))
print('MSE: ', np.mean(mse_total_a))
print('iou: ', np.mean(iou_total_a))

print('Accuracy: ', np.std(acc_total_a))
print('Sensitivity: ', np.std(sensitivity_total_a))
print('specificity: ', np.std(specificity_total_a))
print('precision: ', np.std(precision_total_a))
print('G: ', np.std(G_total_a))
print('F1_score_2: ', np.std(F1_score_2_total_a))
print('MSE: ', np.std(mse_total_a))
print('iou: ', np.std(iou_total_a))

if mode != 'vessel':
    
    print('auc_roc: ', np.mean(auc_roc_total_a))
    print('auc_pr: ', np.mean(auc_pr_total_a))
    print('auc_roc: ', np.std(auc_roc_total_a))
    print('auc_pr: ', np.std(auc_pr_total_a))

#############################################3
print('########################################3')
print('VEIN')
print('#########################################')
#############################################3
print('Accuracy: ', np.mean(acc_total_v))
print('Sensitivity: ', np.mean(sensitivity_total_v))
print('specificity: ', np.mean(specificity_total_v))
print('precision: ', np.mean(precision_total_v))
print('G: ', np.mean(G_total_v))
print('F1_score_2: ', np.mean(F1_score_2_total_v))
print('MSE: ', np.mean(mse_total_v))
print('iou: ', np.mean(iou_total_v))

print('Accuracy: ', np.std(acc_total_v))
print('Sensitivity: ', np.std(sensitivity_total_v))
print('specificity: ', np.std(specificity_total_v))
print('precision: ', np.std(precision_total_v))
print('G: ', np.std(G_total_v))
print('F1_score_2: ', np.std(F1_score_2_total_v))
print('MSE: ', np.std(mse_total_v))
print('iou: ', np.std(iou_total_v))

if mode != 'vessel':
    
    print('auc_roc: ', np.mean(auc_roc_total_v))
    print('auc_pr: ', np.mean(auc_pr_total_v))
    print('auc_roc: ', np.std(auc_roc_total_v))
    print('auc_pr: ', np.std(auc_pr_total_v))

###########################################
print('########################################3')
print('UNCERTAIN')
print('#########################################')
################################################
print('Accuracy: ', np.mean(acc_total_u))
print('Sensitivity: ', np.mean(sensitivity_total_u))
print('specificity: ', np.mean(specificity_total_u))
print('precision: ', np.mean(precision_total_u))
print('G: ', np.mean(G_total_u))
print('F1_score_2: ', np.mean(F1_score_2_total_u))
print('MSE: ', np.mean(mse_total_u))
print('iou: ', np.mean(iou_total_u))

print('Accuracy: ', np.std(acc_total_u))
print('Sensitivity: ', np.std(sensitivity_total_u))
print('specificity: ', np.std(specificity_total_u))
print('precision: ', np.std(precision_total_u))
print('G: ', np.std(G_total_u))
print('F1_score_2: ', np.std(F1_score_2_total_u))
print('MSE: ', np.std(mse_total_u))
print('iou: ', np.mean(iou_total_u))

if mode != 'vessel':
    
    print('auc_roc: ', np.mean(auc_roc_total_u))
    print('auc_pr: ', np.mean(auc_pr_total_u))
    print('auc_roc: ', np.std(auc_roc_total_u))
    print('auc_pr: ', np.std(auc_pr_total_u))

##########################################
print('########################################3')
print('AVERAGE')
print('#########################################')
##########################################
print('Accuracy: ', np.mean(acc_total))
print('Sensitivity: ', np.mean(sensitivity_total))
print('specificity: ', np.mean(specificity_total))
print('precision: ', np.mean(precision_total))
print('G: ', np.mean(G_total))
print('F1_score_2: ', np.mean(F1_score_2_total))
print('MSE: ', np.mean(mse_total))
print('iou: ', np.mean(iou_total))

print('Accuracy: ', np.std(acc_total))
print('Sensitivity: ', np.std(sensitivity_total))
print('specificity: ', np.std(specificity_total))
print('precision: ', np.std(precision_total))
print('G: ', np.std(G_total))
print('F1_score_2: ', np.std(F1_score_2_total))
print('MSE: ', np.std(mse_total))
print('iou: ', np.mean(iou_total))

if mode != 'vessel':
    
    print('auc_roc: ', np.mean(auc_roc_total))
    print('auc_pr: ', np.mean(auc_pr_total))
    print('auc_roc: ', np.std(auc_roc_total))
    print('auc_pr: ', np.std(auc_pr_total))






