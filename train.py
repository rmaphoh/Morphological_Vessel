
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model import UNet, Discriminator, Generator

from torch.utils.tensorboard import SummaryWriter
#from dataset import BasicDataset
from server_code.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split




def train_net(net_G,
              net_D,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              image_size=(592,592),
              save_cp=True,
              ):

    # define path
    img_out_dir="./{}/AV_segmentation_results_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    dir_checkpoint="./{}/checkpoints/AV_model_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    auc_out_dir="{}/AV_auc_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    train_dir= "./data/{}/training/images/".format(args.dataset)
    dir_mask = "./data/{}/training/2st_manual/".format(args.dataset)
    

    # create files
    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.isdir(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    if not os.path.isdir(auc_out_dir):
        os.makedirs(auc_out_dir)

    dataset = BasicDataset(train_dir, dir_mask, image_size)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')
    
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min' if net_G.n_classes > 1 else 'max', factor=0.5, patience=50)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min' if net_G.n_classes > 1 else 'max', factor=0.5, patience=50)
    
    if net_G.n_classes > 1:
        L_seg_CE = nn.CrossEntropyLoss()
    else:
        L_seg_CE = nn.BCEWithLogitsLoss()

    L_seg_MSE = nn.MSELoss()
    L_adv_BCE = nn.BCEWithLogitsLoss()


    for epoch in range(epochs):
        net_G.train()
        net_D.train()
        correct_train = 0
        total_train_pixel = 0
        epoch_loss_G = 0
        epoch_loss_D = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net_G.n_channels, \
                    f'Network has been defined with {net_G.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net_G.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                real_labels = torch.ones(true_masks.size()).to(device=device, dtype=torch.float32)
                fake_labels = torch.zeros(true_masks.size()).to(device=device, dtype=torch.float32)

                '''
                real_labels = torch.ones(batch_size).to(device=device, dtype=torch.float32)
                fake_labels = torch.zeros(batch_size).to(device=device, dtype=torch.float32)
                '''
                #################### train D ##########################
                optimizer_D.zero_grad()

                real_patch = torch.cat([imgs, true_masks], dim=1)

                real_predict_D = net_D(real_patch)
                real_predict_D_sigmoid = torch.sigmoid(real_predict_D)
                #real_predict = net_D(true_masks)

                loss_adv_CE_real = L_adv_BCE(real_predict_D_sigmoid, real_labels)

                loss_adv_CE_real.backward()

                #########################
                
                masks_pred_D = net_G(imgs)

                masks_pred_D_sigmoid = torch.sigmoid(masks_pred_D)

                fake_patch_D = torch.cat([imgs, masks_pred_D_sigmoid], dim=1)

                fake_predict_D = net_D(fake_patch_D)
                fake_predict_D_sigmoid = torch.sigmoid(fake_predict_D)
                #fake_predict = net_D(masks_pred)

                loss_adv_CE_fake = L_adv_BCE(fake_predict_D_sigmoid, fake_labels)

                loss_adv_CE_fake.backward()

                D_Loss = (loss_adv_CE_real + loss_adv_CE_fake)

                

                epoch_loss_D += D_Loss.item()
                writer.add_scalar('Loss/D_train', D_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': D_Loss.item()})

                #optimizer_D.zero_grad()
                #D_Loss.backward()
                #nn.utils.clip_grad_value_(net_D.parameters(), 0.1)
                
                optimizer_D.step()
                
                ################### train G ###########################
                optimizer_G.zero_grad()

                masks_pred_G = net_G(imgs)

                masks_pred_G_sigmoid = torch.sigmoid(masks_pred_G)

                fake_patch_G = torch.cat([imgs, masks_pred_G_sigmoid], dim=1)
            
                #fake_predict = net_D(masks_pred)
                fake_predict_G = net_D(fake_patch_G)
                fake_predict_G_sigmoid = torch.sigmoid(fake_predict_G)

                loss_adv_G_fake = L_adv_BCE(fake_predict_G_sigmoid, real_labels)

                #masks_pred_sigmoid = torch.sigmoid(masks_pred)
                
                loss_seg_CE = L_seg_CE(masks_pred_G, true_masks)
                loss_seg_MSE = L_seg_MSE(masks_pred_G_sigmoid, true_masks)

                G_Loss = 0.07*loss_adv_G_fake + 1.1*loss_seg_CE + 0.3*loss_seg_MSE
                #G_Loss = 0.08*loss_adv_G_fake 
                

                epoch_loss_G += G_Loss.item()
                writer.add_scalar('Loss/G_train', G_Loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': G_Loss.item()})

                #optimizer_G.zero_grad()
                G_Loss.backward()
                #nn.utils.clip_grad_value_(net_G.parameters(), 0.1)
                optimizer_G.step()

                ##########################################################

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (1 * batch_size)) == 0:
                    for tag, value in net_G.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score, acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr = eval_net(epoch, net_G, val_loader, device, mask=True)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer_G.param_groups[0]['lr'], global_step)

                    logging.info('Validation sensitivity: {}'.format(sensitivity))
                    writer.add_scalar('sensitivity/val_G', sensitivity, global_step)

                    logging.info('Validation specificity: {}'.format(specificity))
                    writer.add_scalar('specificity/val_G', specificity, global_step)

                    logging.info('Validation precision: {}'.format(precision))
                    writer.add_scalar('precision/val_G', precision, global_step)

                    logging.info('Validation G: {}'.format(G))
                    writer.add_scalar('G/val_G', G, global_step)

                    logging.info('Validation F1_score: {}'.format(F1_score_2))
                    writer.add_scalar('F1_score/val_G', F1_score_2, global_step)

                    logging.info('Validation auc_roc: {}'.format(auc_roc))
                    writer.add_scalar('Auc_roc/val_G', auc_roc, global_step)

                    logging.info('Validation auc_pr: {}'.format(auc_pr))
                    writer.add_scalar('Auc_pr/val_G', auc_pr, global_step)

                    if net_G.n_classes > 1:
                        prediction_binary = (torch.sigmoid(masks_pred_G) > 0.5)
                        prediction_binary_gpu = prediction_binary.to(device=device, dtype=mask_type)
                        # write accuracy
                        correct_train += prediction_binary_gpu.eq(true_masks.data).sum().item()
                        total_train_pixel += prediction_binary_gpu.nelement()
                        train_accuracy = 100 * correct_train / total_train_pixel
                        
                        logging.info('Validation accuracy: {}'.format(train_accuracy))
                        writer.add_scalar('Acc/test_G', train_accuracy, global_step)
                    else:
                        # G
                        prediction_binary = (torch.sigmoid(masks_pred_G) > 0.5)
                        prediction_binary_gpu = prediction_binary.to(device=device, dtype=mask_type)
                        # write accuracy

                        correct_train += prediction_binary_gpu.eq(true_masks.data).sum().item()
                        total_train_pixel += prediction_binary_gpu.nelement()
                        train_accuracy = 100 * correct_train / total_train_pixel
                        
                        logging.info('Validation accuracy: {}'.format(train_accuracy))
                        writer.add_scalar('Acc/val_G', train_accuracy, global_step)

                        # D
                        real_predict_binary = (torch.sigmoid(real_predict_D) > 0.5)
                        real_predict_binary_gpu = real_predict_binary.to(device=device, dtype=mask_type)

                        fake_predict_binary = (torch.sigmoid(fake_predict_D) > 0.5)
                        fake_predict_binary_gpu = fake_predict_binary.to(device=device, dtype=mask_type)

                        prediction_binary_DR = real_predict_binary_gpu.eq(real_labels.data).sum().item()
                        prediction_binary_DF = fake_predict_binary_gpu.eq(fake_labels.data).sum().item()

                        aver_prediction_binary_D = (prediction_binary_DR + prediction_binary_DF)/2
                        train_accuracy_D = 100 * aver_prediction_binary_D / total_train_pixel
                        logging.info('Validation accuracy: {}'.format(train_accuracy_D))
                        writer.add_scalar('Acc/Val_D', train_accuracy_D, global_step)
                        
                        correct_train = 0
                        total_train_pixel = 0
                    
                    if net_G.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test_G', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test_G', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net_G.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred_G) > 0.5, global_step)

                        
        
        if epoch%20==0:
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net_G.state_dict(),
                        dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=240,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=6,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    parser.add_argument( '-r', '--ratio_gan2seg', metavar='G2S', type=int, default=10,
                        help='ratio of gan loss to seg loss', dest='gan2seg')
    parser.add_argument( '-dis', '--discriminator', dest='dis', type=str, default=False,
                        help='type of discriminator')
    parser.add_argument( '-d','--dataset', dest='dataset', type=str, 
                        help='dataset name')
    parser.add_argument( '-v', '--validation', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')


    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    alpha_recip=1./args.gan2seg if args.gan2seg>0 else 0

    dataset=args.dataset
    img_size= (592,880) if dataset=='HRF' else (592,592) 

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    net_G = Generator(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)

    net_D = Discriminator(input_channels=4, n_filters = 32, n_classes=1, bilinear=False)



    logging.info(f'Network_G:\n'
                 f'\t{net_G.n_channels} input channels\n'
                 f'\t{net_G.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net_G.bilinear else "Transposed conv"} upscaling')

    logging.info(f'Network_D:\n'
                 f'\t{net_D.n_channels} input channels\n'
                 f'\t{net_D.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net_D.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net_G.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
        net_D.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net_G.to(device=device)
    net_D.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net_G=net_G,
                  net_D=net_D,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  image_size=img_size)
    except KeyboardInterrupt:
        torch.save(net_G.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
















'''



import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model import UNet

from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split



def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              image_size=(592,592),
              save_cp=True,
              ):

    # define path
    img_out_dir="./{}/AV_segmentation_results_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    dir_checkpoint="./{}/checkpoints/AV_model_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    auc_out_dir="{}/AV_auc_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    train_dir= "./data/{}/training/images/".format(args.dataset)
    dir_mask = "./data/{}/training/2st_manual/".format(args.dataset)
    

    # create files
    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.isdir(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    if not os.path.isdir(auc_out_dir):
        os.makedirs(auc_out_dir)

    dataset = BasicDataset(train_dir, dir_mask, image_size)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0


    ##############################
    ##############################
    
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', factor=0.5, patience=200)
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, 
                    f'Network has been defined with {net.n_channels} input channels, ' 
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' 
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (1 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        if epoch%10==0:
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                        dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=240,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=6,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    parser.add_argument( '-r', '--ratio_gan2seg', metavar='G2S', type=int, default=10,
                        help='ratio of gan loss to seg loss', dest='gan2seg')
    parser.add_argument( '-dis', '--discriminator', dest='dis', type=str, default=False,
                        help='type of discriminator')
    parser.add_argument( '-d','--dataset', dest='dataset', type=str, 
                        help='dataset name')
    parser.add_argument( '-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')


    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    alpha_recip=1./args.gan2seg if args.gan2seg>0 else 0

    dataset=args.dataset
    img_size= (592,880) if dataset=='HRF' else (592,592) 

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  image_size=img_size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

'''
