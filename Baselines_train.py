import os
import glob
from os import listdir

import logging

from PIL import Image
from os.path import splitext
import sys
import random
import logging
import errno
import torch
import timeit

import imageio


import argparse

import tqdm

from PIL import Image, ImageEnhance

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

from scipy.ndimage import rotate

# from torch.utils.tensorboard import SummaryWriter

from tensorboardX import SummaryWriter

from torchvision import transforms
# ====================================

from dataset import BasicDataset
from torch.utils.data import Dataset, DataLoader, random_split

# ====================================

from Baselines_models import MTSARVSnet, UNet, DualAttUNet
# from Baselines_metrics import segmentation_scores, f1_score

# from eval import pad_imgs, dice_coeff, AUC_PR, AUC_ROC, pixel_values_in_mask, misc_measures
# from sklearn.metrics import auc, confusion_matrix, roc_auc_score, precision_recall_curve

from Baselines_metrics import eval_net_multitask

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    #transforms.Resize(256),
    #transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.Pad((13,4,14,4), fill=0, padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(360),
    transforms.ToTensor(),
    #normalize
])

val_transformer = transforms.Compose([
    #transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.Pad((13,4,14,4), fill=0, padding_mode='constant'),
    transforms.ToTensor(),
    #normalize
])


class MultiTaskDataset(Dataset):

    def __init__(self, imgs_dir, masks_dir_main, masks_dir_auxilary, roi_mask_dir, img_size, train_or, mask_suffix=''):

        self.imgs_dir = imgs_dir
        self.masks_dir_main = masks_dir_main
        self.masks_dir_auxilary = masks_dir_auxilary
        self.roi_mask_dir = roi_mask_dir

        self.train_or = train_or

        self.mask_suffix = mask_suffix
        self.img_size = img_size
        # self.transform = transforms

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {(self.ids)} ')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs

    @classmethod
    def pad_imgs(self, imgs, img_size):
        img_h, img_w = imgs.shape[0], imgs.shape[1]
        target_h, target_w = img_size[0], img_size[1]
        if len(imgs.shape) == 3:
            d = imgs.shape[2]
            padded = np.zeros((target_h, target_w, d))
        elif len(imgs.shape) == 2:
            padded = np.zeros((target_h, target_w))
        padded[(target_h - img_h) // 2:(target_h - img_h) // 2 + img_h, (target_w - img_w) // 2:(target_w - img_w) // 2 + img_w, ...] = imgs
        # print(np.shape(padded))
        return padded

    @classmethod
    def preprocess(self, pil_img, mask1, mask2, roi, img_size, train_or, k):
        # w, h = pil_img.size
        newW, newH = img_size[0], img_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_array = np.array(pil_img)
        roi = np.array(roi)
        mask_array1 = np.array(mask1) / 255
        mask_array2 = np.array(mask2) / 255

        img_array = self.pad_imgs(img_array, img_size)
        mask_array1 = self.pad_imgs(mask_array1, img_size)
        mask_array2 = self.pad_imgs(mask_array2, img_size)

        roi = self.pad_imgs(roi, img_size)

        # print('@@@@@@@@@@@@@@', np.shape(img_array))
        # print('@@@@@@@@@@@@@@', np.shape(mask_array1))
        # print('@@@@@@@@@@@@@@', np.shape(mask_array2))

        if train_or:
            flipping_dice = np.random.random()
            if flipping_dice <= 0.25:
                # img_array = img_array[:, ::-1, :]  # flipped imgs
                # mask_array1 = mask_array1[:, ::-1, :]
                # mask_array2 = mask_array2[:, ::-1]
                # flip x:
                img_array = np.flip(img_array, axis=0).copy()
                roi = np.flip(roi, axis=0).copy()
                mask_array1 = np.flip(mask_array1, axis=0).copy()
                mask_array2 = np.flip(mask_array2, axis=0).copy()
            elif flipping_dice <= 0.5:
                img_array = np.flip(img_array, axis=1).copy()
                mask_array1 = np.flip(mask_array1, axis=1).copy()
                mask_array2 = np.flip(mask_array2, axis=1).copy()
                roi = np.flip(roi, axis=1).copy()
            elif flipping_dice <= 0.75:
                img_array = np.flip(img_array, axis=1).copy()
                roi = np.flip(roi, axis=1).copy()
                mask_array1 = np.flip(mask_array1, axis=1).copy()
                mask_array2 = np.flip(mask_array2, axis=1).copy()
                img_array = np.flip(img_array, axis=0).copy()
                mask_array1 = np.flip(mask_array1, axis=0).copy()
                mask_array2 = np.flip(mask_array2, axis=0).copy()
                roi = np.flip(roi, axis=0).copy()

            angle = 3 * np.random.randint(120)
            img_array = rotate(img_array, angle, axes=(0, 1), reshape=False)
            roi = rotate(roi, angle, axes=(0, 1), reshape=False)
            # print('@@@@@@@@@@@@@@', np.shape(img_array))
            # print('@@@@@@@@@@@@@@', np.shape(mask_array))

            img_array = self.random_perturbation(img_array)
            mask_array1 = np.round(rotate(mask_array1, angle, axes=(0, 1), reshape=False))
            mask_array2 = np.round(rotate(mask_array2, angle, axes=(0, 1), reshape=False))

        # if train_or:
        #     if np.random.random() > 0.5:
        #         img_array = np.flip(img_array, axis=0).copy()
        #         mask_array1 =

        mean_r = np.mean(img_array[..., 0][img_array[..., 0] > 00.0], axis=0)
        std_r = np.std(img_array[..., 0][img_array[..., 0] > 00.0], axis=0)

        mean_g = np.mean(img_array[..., 1][img_array[..., 0] > 00.0], axis=0)
        std_g = np.std(img_array[..., 1][img_array[..., 0] > 00.0], axis=0)

        mean_b = np.mean(img_array[..., 2][img_array[..., 0] > 00.0], axis=0)
        std_b = np.std(img_array[..., 2][img_array[..., 0] > 00.0], axis=0)
        # print('!!!!!!!!!!!', len(mean))
        # print('!!!!!!!!!!!', len(std))

        # assert len(mean)==3 and len(std)==3
        # img_array=(img_array-mean)/std
        img_array[..., 0] = (img_array[..., 0] - mean_r) / std_r
        img_array[..., 1] = (img_array[..., 1] - mean_g) / std_g
        img_array[..., 2] = (img_array[..., 2] - mean_b) / std_b

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)

        if len(mask_array1.shape) == 2:
            mask_array1 = np.expand_dims(mask_array1, axis=2)

        if len(mask_array2.shape) == 2:
            mask_array2 = np.expand_dims(mask_array2, axis=2)

        if len(roi.shape) == 2:
            roi = np.expand_dims(roi, axis=2)
        # print(np.shape(img_array))

        # image_array_img = Image.fromarray((img_array*255).astype(np.uint8))
        # image_array_img.save('./aug_results/new/inside_img_{:02}.png'.format(k))
        # mask_array_img_squ = np.squeeze(mask_array)
        # mask_array_img_squ = Image.fromarray((mask_array_img_squ*255).astype(np.uint8))
        # image_array_img.save('./aug_results/new/inside_img_{:02}.png'.format(k))
        # mask_array_img_squ.save('./aug_results/new/inside_mask_{:02}.png'.format(k))
        img_array = img_array.transpose((2, 0, 1))

        # plt.imshow(mask_array1)
        # plt.show()

        mask_array1 = mask_array1.transpose((2, 0, 1))
        mask_array1 = np.where(mask_array1 > 0.5, 1.0, 0.0)
        #
        mask_array2 = mask_array2.transpose((2, 0, 1))
        mask_array2 = np.where(mask_array2 > 0.5, 1.0, 0.0)

        roi = roi.transpose((2, 0, 1))
        roi = np.where(roi > 0.5, 1.0, 0.0)
        # print('!!!!!!!!!!!!!!', np.shape(img_array))
        # print('!!!!!!!!!!!!!!', np.shape(mask_array1))
        # print('!!!!!!!!!!!!!!', np.shape(mask_array2))

        c, h, w = mask_array1.shape[0], mask_array1.shape[1], mask_array1.shape[2]

        # (todo) two dimension (h, w)
        # mask_array1_new = np.zeros((h, w), dtype=np.float32)
        mask_array1_new = np.zeros((1, h, w), dtype=np.float32)

        # for hh in range(h):
        #     for ww in range(w):
        #         if mask_array1[hh, ww, 0] == 0 and mask_array1[hh, ww, 1] == 0 and mask_array1[hh, ww, 2] == 0:
        #             print('yes')
        #         else:
        #             print('no')

        # print(np.unique(mask_array1))
        # print(np.unique(mask_array2))

        # black_id = np.where(np.logical_and(mask_array1[:, :, 0] == 0.0, mask_array1[:, :, 1] == 0.0, mask_array1[:, :, 2] == 0.0))
        # print(len(black_id[0]))
        #
        # blue_id = np.where(np.logical_and(mask_array1[:, :, 0] == 0.0, mask_array1[:, :, 1] == 0.0, mask_array1[:, :, 2] == 0.0))
        # print(len(blue_id[0]))

        # transform mask array into 0, 1, 2, 3 for back ground, artery, vein and uncertain
        black_id = np.where(np.logical_and(np.logical_and(mask_array1[0, :, :] == 0.0, mask_array1[1, :, :] == 0.0).astype('float32'), mask_array1[2, :, :] == 0.0))
        red_id = np.where(np.logical_and(np.logical_and(mask_array1[0, :, :] == 1.0, mask_array1[1, :, :] == 0.0).astype('float32'), mask_array1[2, :, :] == 0.0))
        blue_id = np.where(np.logical_and(np.logical_and(mask_array1[0, :, :] == 0.0, mask_array1[1, :, :] == 0.0).astype('float32'), mask_array1[2, :, :] == 1.0))
        green_id = np.where(np.logical_and(np.logical_and(mask_array1[0, :, :] == 0.0, mask_array1[1, :, :] == 1.0).astype('float32'), mask_array1[2, :, :] == 0.0))
        white_id = np.where(np.logical_and(np.logical_and(mask_array1[0, :, :] == 1.0, mask_array1[1, :, :] == 1.0).astype('float32'), mask_array1[2, :, :] == 1.0))

        mask_array1_new[0, :, :][black_id] = 0.0
        mask_array1_new[0, :, :][red_id] = 1.0
        mask_array1_new[0, :, :][blue_id] = 2.0
        mask_array1_new[0, :, :][green_id] = 3.0
        mask_array1_new[0, :, :][white_id] = 3.0

        # black_id = np.where(mask_array1_new[:, :] == 0.0)
        # print(len(black_id[0]))
        #
        # red_id = np.where(mask_array1_new[:, :] == 1.0)
        # print(len(red_id[0]))
        #
        # blue_id = np.where(mask_array1_new[:, :] == 2.0)
        # print(len(blue_id[0]))

        k += 1

        # print(np.unique(mask_array1_new))

        # for hh in range(h):
        #     for ww in range(w):
        #         if mask_array1[hh, ww, 0] == 0 and mask_array1[hh, ww, 1] == 0 and mask_array1[hh, ww, 2] == 0:
        #             print('yes')
        #         else:
        #             print('no')

        # mask_array1_new_temp = mask_array1_new[mask_array1_new == 1]
        # print(np.shape(mask_array1_new_temp))

        # print(len(mask_array1_new == 1))

        # print(np.shape(mask_array1_new))
        # print(np.shape(mask_array2))
        # print(np.unique(mask_array2))

        return img_array, mask_array1_new, mask_array2, roi

    def __getitem__(self, index):

        # idx = self.ids[i]

        # print(idx)
        # print(self.masks_dir_main + idx + '.*')
        # print(self.masks_dir_auxilary + idx + '.*')

        # all_images = glob.glob(os.path.join(self.imgs_dir, '*.png'))
        # all_labels = glob.glob(os.path.join(self.labels_folder, '*.png'))
        #
        # all_labels.sort()
        # all_images.sort()

        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        # mask_file_main = glob(self.masks_dir_main + idx + '.png')
        # mask_file_auxilary = glob(self.masks_dir_auxilary + idx + '.png')
        # logging.info(f'Creating dataset with {len(mask_file)} mask')

        # img_file = glob(self.imgs_dir + idx + '.png')

        # assert len(mask_file_main) == 1, \
        #     f'Either no main mask or multiple masks found for the ID {idx}: {mask_file_main}'
        # assert len(mask_file_auxilary) == 1, \
        #     f'Either no auxilary mask or multiple masks found for the ID {idx}: {mask_file_auxilary}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {img_file}'

        # print(os.path.join(self.imgs_dir, '*.png'))

        all_images = glob.glob(os.path.join(self.imgs_dir, '*.tif'))
        all_labels_main = glob.glob(os.path.join(self.masks_dir_main, '*.png'))
        all_labels_auxilary = glob.glob(os.path.join(self.masks_dir_auxilary, '*.png'))

        all_rois = glob.glob(os.path.join(self.roi_mask_dir, '*.gif'))

        # print(len(all_images))
        # print(len(all_labels_main))
        # print(len(all_labels_auxilary))

        all_labels_main.sort()
        all_labels_auxilary.sort()
        all_images.sort()
        all_rois.sort()

        # mask_main = Image.open(mask_file_main[0])
        # mask_auxilary = Image.open(mask_file_auxilary[0])
        # img = Image.open(img_file[0])

        mask_main = Image.open(all_labels_main[index])
        mask_auxilary = Image.open(all_labels_auxilary[index])
        img = Image.open(all_images[index])

        roi = Image.open(all_rois[index])

        # assert img.size == mask_main.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask_main.size}'

        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        # torch.cuda.manual_seed(seed)
        #
        # if self.transform:
        #     img = self.transform(img)
        #
        # # random.seed(seed)  # apply this seed to target tranfsorms
        # # torch.cuda.manual_seed(seed)  # needed for torchvision 0.7
        #
        # if self.transform:
        #     mask_main = self.transform(mask_main)
        #     mask_auxilary = self.transform(mask_auxilary)
        #
        # img = self.preprocess(img, self.img_size)
        # mask_main = self.preprocess(mask_main, self.img_size)
        # mask_auxilary = self.preprocess(mask_auxilary, self.img_size)
        # if self.transform:
        img, mask_main, mask_auxilary, roi = self.preprocess(img, mask_main, mask_auxilary, roi, self.img_size, self.train_or, index)
        # print('the range of img is: ',np.unique(img))
        # print('the range of mask is: ',np.unique(mask))
        '''
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        '''
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask_main': torch.from_numpy(mask_main).type(torch.FloatTensor),
            'mask_auxilary': torch.from_numpy(mask_auxilary).type(torch.FloatTensor),
            'roi': torch.from_numpy(roi).type(torch.FloatTensor)
        }


def trainModels(dataset_name,
                data_directory,
                input_dim,
                class_no,
                repeat,
                train_batchsize,
                num_epochs,
                learning_rate,
                network,
                log_tag,
                save_threshold_epoch=400,
                save_interval_epoch=10,
                image_size=(592, 592),
                multi_task=True):
    #
    for j in range(1, repeat + 1):
        #
        repeat_str = str(j)
        #
        if network == 'MTSARVSnet':

            assert multi_task is True

            Exp = MTSARVSnet(input_dim=input_dim, output_dim=class_no)
            #
            Exp_name = 'MTSARVSnet_' + \
                       '_train_batch_' + str(train_batchsize) + \
                       '_repeat_' + str(repeat_str) + \
                       '_lr_' + str(learning_rate) + \
                       '_total_epoch_' + str(num_epochs) + \
                       '_' + dataset_name

        elif network == 'unet':

            Exp = UNet(in_ch=input_dim, width=32, class_no=class_no)
            #
            Exp_name = 'Unet_' + \
                       '_train_batch_' + str(train_batchsize) + \
                       '_repeat_' + str(repeat_str) + \
                       '_lr_' + str(learning_rate) + \
                       '_total_epoch_' + str(num_epochs) + \
                       '_' + dataset_name

        elif network == 'dual_attention_unet':

            Exp = DualAttUNet(in_ch=input_dim, width=32, class_no=class_no)
            #
            Exp_name = 'Dual_Attention_Unet_' + \
                       '_train_batch_' + str(train_batchsize) + \
                       '_repeat_' + str(repeat_str) + \
                       '_lr_' + str(learning_rate) + \
                       '_total_epoch_' + str(num_epochs) + \
                       '_' + dataset_name

        # ====================================================================================================================================================================
        trainloader, validateloader, testloader = getData(data_directory, dataset_name, train_batchsize, image_size, multi_task)
        # ==================
        trainSingleModel(Exp,
                         Exp_name,
                         num_epochs,
                         learning_rate,
                         dataset_name,
                         trainloader,
                         validateloader,
                         testloader,
                         log_tag,
                         class_no,
                         save_threshold_epoch,
                         save_interval_epoch)


def getData(data_directory, dataset_name, train_batchsize, image_size, multi_task=True):

    # img_out_dir="./{}/AV_segmentation_results_{}_{}".format(args.dataset, args.dis, args.gan2seg)
    # dir_checkpoint="./{}/checkpoints/AV_model_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    # auc_out_dir="{}/AV_auc_{}_{}".format(args.dataset, args.dis, args.gan2seg)

    if multi_task is False:

        train_dir = data_directory + '/' + dataset_name + '/training/images/'
        dir_mask = data_directory + '/' + dataset_name + '/training/1st_manual/'

        test_dir_img = data_directory + '/' + dataset_name + '/test/images/'
        test_dir_mask = data_directory + '/' + dataset_name + '/test/1st_manual/'

    else:

        train_dir = data_directory + '/' + dataset_name + '/training/images/'
        dir_mask_main = data_directory + '/' + dataset_name + '/training/1st_manual/'
        dir_mask_auxiliary = data_directory + '/' + dataset_name + '/training/2st_manual/'

        test_dir_img = data_directory + '/' + dataset_name + '/test/images/'
        test_dir_mask_main = data_directory + '/' + dataset_name + '/test/1st_manual/'
        test_dir_mask_auxilary = data_directory + '/' + dataset_name + '/test/2nd_manual/'

        test_dir_roi = data_directory + '/' + dataset_name + '/test/mask/'

    # train_dir = "./data/DRIVE_AV/training/images/"
    # dir_mask = "./data/DRIVE_AV/training/2st_manual/"

    # # create files
    # if not os.path.isdir(img_out_dir):
    #     os.makedirs(img_out_dir)
    # if not os.path.isdir(dir_checkpoint):
    #     os.makedirs(dir_checkpoint)
    # if not os.path.isdir(auc_out_dir):
    #     os.makedirs(auc_out_dir)

    # val_percent = 0.1

    if multi_task is False:

        dataset = BasicDataset(train_dir, dir_mask, image_size)
        test_dataset = BasicDataset(test_dir_img, test_dir_mask, image_size)

    else:

        dataset = MultiTaskDataset(train_dir, dir_mask_main, dir_mask_auxiliary, test_dir_roi, image_size, train_or=True)
        test_dataset = MultiTaskDataset(test_dir_img, test_dir_mask_main, test_dir_mask_auxilary, test_dir_roi, image_size, train_or=False)

    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val

    n_val = 1
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=train_batchsize, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     datasettag,
                     trainloader,
                     validateloader,
                     testloader,
                     log_tag,
                     class_no,
                     save_threshold,
                     save_interval):

    #
    device = torch.device('cuda')
    #
    save_model_name = model_name
    #
    saved_information_path = './Results'
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + datasettag
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + log_tag
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_log_path = saved_information_path + '/Logs'
    #
    try:
        os.mkdir(saved_log_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + save_model_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('The current model is:')

    print(save_model_name)

    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)
    # writer = SummaryWriter(comment=f'LR_{learning_rate}')

    # logging.info(f'''Starting training:
    #     Epochs:          {num_epochs}
    #     Learning rate:   {learning_rate}
    # ''')

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):
        #
        model.train()
        #
        train_h_dists = 0
        train_f1 = 0
        train_iou = 0

        train_main_loss = 0
        train_auxilar_loss = 0
        train_total_loss = 0

        train_recall = 0
        train_precision = 0
        train_effective_h = 0

        # for j, (images, labels) in enumerate(trainloader):
        for j, batch_current in enumerate(trainloader):

            # print(np.unique(labels))

            if 'MTSARVSnet' in model_name:

                images = batch_current['image']
                labels_main = batch_current['mask_main']
                labels_auxilary = batch_current['mask_auxilary']

                # print(np.unique(labels_main))
                # print(np.unique(labels_auxilary))

                # print(np.shape(labels_main))
                # print(np.shape(labels_auxilary))

                images = images.to(device=device, dtype=torch.float32)
                labels_main = labels_main.to(device=device, dtype=torch.long)
                labels_auxilary = labels_auxilary.to(device=device, dtype=torch.float32)

            else:

                images = batch_current['image']
                labels = batch_current['mask_main']
                labels_auxilary = batch_current['mask_auxilary']

                # print(np.unique(labels))

                images = images.to(device=device, dtype=torch.float32)
                # labels = labels.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

            optimizer.zero_grad()

            if 'MTSARVSnet' in model_name:

                output, side_output1, side_output2, side_output3, output_v, side_output1_v, side_output2_v, side_output3_v = model(images)

                # print(output.size())

                _, prob_outputs = torch.max(output, dim=1)
                _, prob_outputs_side1 = torch.max(side_output1, dim=1)
                _, prob_outputs_side2 = torch.max(side_output2, dim=1)
                _, prob_outputs_side3 = torch.max(side_output3, dim=1)

                # sideloss1 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output1, dim=1), labels.squeeze(1))
                # sideloss2 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output2, dim=1), labels.squeeze(1))
                # sideloss3 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output3, dim=1), labels.squeeze(1))

                # sideloss1 = nn.BCELoss(reduction='mean')(side_output1, labels_main)
                # sideloss2 = nn.BCELoss(reduction='mean')(side_output2, labels_main)
                # sideloss3 = nn.BCELoss(reduction='mean')(side_output3, labels_main)
                # mainloss_main = nn.BCELoss(reduction='mean')(output, labels_main)

                mainloss_main = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(output, dim=1), labels_main.squeeze(1))
                sideloss1 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output1, dim=1), labels_main.squeeze(1))
                sideloss2 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output2, dim=1), labels_main.squeeze(1))
                sideloss3 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output3, dim=1), labels_main.squeeze(1))

                auxilary_loss_main = nn.BCELoss(reduction='mean')(output_v, labels_auxilary)
                auxilary_loss_side1 = nn.BCELoss(reduction='mean')(side_output1_v, labels_auxilary)
                auxilary_loss_side2 = nn.BCELoss(reduction='mean')(side_output2_v, labels_auxilary)
                auxilary_loss_side3 = nn.BCELoss(reduction='mean')(side_output3_v, labels_auxilary)

                auxilary_loss = auxilary_loss_main + (auxilary_loss_side1 + auxilary_loss_side2 + auxilary_loss_side3) / 3

                main_loss = mainloss_main + (sideloss1 + sideloss2 + sideloss3) / 3

                loss = main_loss + auxilary_loss

                train_total_loss += loss
                train_main_loss += main_loss
                train_auxilar_loss += auxilary_loss

            else:

                output = model(images)
                loss = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(output, dim=1), labels.squeeze(1))
                train_total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # class_outputs = prob_outputs
        # old version
        # val_score, acc, sensitivity, specificity, precision, G, F1_score_2 = eval_net_multitask(epoch, model, validateloader, device, mask=True, mode='vessel', model_name=model_name)
        # current version:
        accuracy_eva, iou_eva, precision_eva, recall_eva, f1_eva, specificity_eva, sensitivity_eva, g_eva, auc_pr_eva, auc_roc_eva, mse_eva = eval_net_multitask(epoch, model, validateloader, device, mask=True, mode='vessel', model_name=model_name)

        if 'MTSARVSnet' in model_name:
            print(
                'Step [{}/{}], '
                'Train total loss: {:.4f}, '
                'Train auxilary loss: {:.4f}, '
                'Train main loss: {:.4f}, '
                'val acc: {:.4f}, '
                'val iou: {:.4f}, '
                'val F1:{:.4f}, '.format(epoch + 1, num_epochs,
                                         train_total_loss / (j + 1),
                                         train_auxilar_loss / (j + 1),
                                         train_main_loss / (j + 1),
                                         accuracy_eva,
                                         iou_eva,
                                         f1_eva))
        else:
            print(
                'Step [{}/{}], '
                'Train total loss: {:.4f}, '
                'val acc: {:.4f}, '
                'val iou: {:.4f}, '
                'val F1:{:.4f}, '.format(epoch + 1, num_epochs,
                                         train_total_loss / (j + 1),
                                         accuracy_eva,
                                         iou_eva,
                                         f1_eva))

        # logging.info('Validation sensitivity: {}'.format(sensitivity))
        # writer.add_scalar('sensitivity/val', sensitivity, epoch)

        # logging.info('Validation specificity: {}'.format(specificity))
        # writer.add_scalar('specificity/val', specificity, epoch)

        logging.info('Validation precision: {}'.format(precision_eva))
        writer.add_scalar('precision/val', precision_eva, epoch)

        logging.info('Validation recall: {}'.format(recall_eva))
        writer.add_scalar('recall/val', recall_eva, epoch)

        logging.info('Validation iou: {}'.format(iou_eva))
        writer.add_scalar('iou/val', iou_eva, epoch)
        # logging.info('Validation: {}'.format(G))
        # writer.add_scalar('G/val', G, epoch)

        logging.info('Validation F1_score: {}'.format(f1_eva))
        writer.add_scalar('F1_score/val', f1_eva, epoch)

        # if (class_outputs == 1).sum() > 1 and (labels == 1).sum() > 1:
        #     #
        #     dist_ = hd95(class_outputs, labels, class_no)
        #     train_h_dists += dist_
        #     train_effective_h = train_effective_h + 1
        #
        # train_mean_iu_ = segmentation_scores(labels, class_outputs, class_no)
        # #
        # train_f1_, train_recall_, train_precision_, TPs_, TNs_, FPs_, FNs_, Ps_, Ns_ = f1_score(labels, class_outputs, class_no)
        #
        # train_main_loss += loss.item()
        # train_f1 += train_f1_
        # train_iou += train_mean_iu_
        # train_recall += train_recall_
        # train_precision += train_precision_

        # # Evaluate at the end of each epoch:
        # model.eval()
        # with torch.no_grad():
        #     #
        #     validate_iou = 0
        #     validate_f1 = 0
        #     validate_h_dist = 0
        #     validate_h_dist_effective = 0
        #     #
        #     # for i, (val_img, val_label, val_name) in enumerate(validateloader):
        #     for i, val_batch_current in enumerate(validateloader):
        #
        #         # val_img = val_batch_current['image']
        #         # val_label = val_batch_current['mask']
        #         #
        #         # val_img = val_img.to(device=device, dtype=torch.float32)
        #         # val_label = val_label.to(device=device, dtype=torch.long)
        #
        #         if 'MTSARVSnet' in model_name:
        #
        #             val_img = batch_current['image']
        #             labels_main = batch_current['mask_main']
        #             labels_auxilary = batch_current['mask_auxilary']
        #
        #             images = images.to(device=device, dtype=torch.float32)
        #             labels_main = labels_main.to(device=device, dtype=torch.float32)
        #             labels_auxilary = labels_auxilary.to(device=device, dtype=torch.float32)
        #
        #         else:
        #
        #             images = batch_current['image']
        #             labels = batch_current['mask']
        #
        #             images = images.to(device=device, dtype=torch.float32)
        #             labels = labels.to(device=device, dtype=torch.float32)
        #
        #
        #         if 'MTSARVSnet' in model_name:
        #
        #             val_outputs, val_outputs_side1, val_outputs_side2, val_outputs_side3 = model(val_img)
        #             _, val_class_outputs = torch.max(val_outputs, dim=1)
        #
        #         eval_mean_iu_ = segmentation_scores(val_label, val_class_outputs, class_no)
        #         eval_f1_, eval_recall_, eval_precision_, eTP, eTN, eFP, eFN, eP, eN = f1_score(val_label, val_class_outputs, class_no)
        #         validate_iou += eval_mean_iu_
        #         validate_f1 += eval_f1_
        #         #
                # if (val_class_outputs == 1).sum() > 1 and (val_label == 1).sum() > 1:
                #     v_dist_ = hd95(val_class_outputs, val_label, class_no)
                #     validate_h_dist += v_dist_
                #     validate_h_dist_effective = validate_h_dist_effective + 1


        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #
        # writer.add_scalars('acc metrics', {'train iou': train_iou / (j+1),
        #                                    'train hausdorff dist': train_h_dists / (train_effective_h+1),
        #                                    'val iou': validate_iou / (i + 1),
        #                                    'val f1': validate_f1 / (i + 1)}, epoch + 1)
        #
        # writer.add_scalars('loss values', {'main loss': train_main_loss / (j+1)}, epoch + 1)

        for param_group in optimizer.param_groups:
            #
            if epoch == (num_epochs - 10):
                param_group['lr'] = learning_rate * 0.1
                # param_group['lr'] = learning_rate * ((1 - epoch / num_epochs) ** 0.999)

        if epoch > save_threshold:

            if epoch % save_interval == 0:

                dir_checkpoint = saved_information_path + '/CheckPoints'

                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass

                torch.save(model, dir_checkpoint + '/' + save_model_name + '_current_epoch' + str(epoch) + '.pth')

                print(save_model_name + '_' + str(epoch) + 'saved')

    # # ===============
    # Testing:
    # =================
    save_path = saved_information_path + '/Visual_results'

    try:

        os.mkdir(save_path)

    except OSError as exc:

        if exc.errno != errno.EEXIST:

            raise

        pass
    #
    # model.eval()
    # with torch.no_grad():
    #
    #     test_iou = 0
    #     test_f1 = 0
    #     test_h_dist = 0
    #     test_h_dist_effective = 0
    #     #
    #     for ii, (test_img, test_label, test_name) in enumerate(testdata):
    #         #
    #         test_img = test_img.to(device=device, dtype=torch.float32)
    #         test_label = test_label.to(device=device, dtype=torch.float32)
    #
    #         assert torch.max(test_label) == 1.0
    #         assert torch.min(test_label) == 0.0
    #
    #         if 'ERF' in model_name:
    #             #
    #             test_outputs_fp, test_outputs_fn = model(test_img)
    #             test_outputs_fp = torch.sigmoid(test_outputs_fp)
    #             test_outputs_fn = torch.sigmoid(test_outputs_fn)
    #             #
    #             inverse_test_outputs_fn = torch.ones_like(test_outputs_fn).to(device=device, dtype=torch.float32)
    #             inverse_test_outputs_fn = inverse_test_outputs_fn - test_outputs_fn
    #             test_class_outputs = (inverse_test_outputs_fn + test_outputs_fp) / 2
    #             #
    #         else:
    #             test_outputs = model(test_img)
    #             test_class_outputs = torch.sigmoid(test_outputs)
    #             #
    #         test_mean_iu_ = segmentation_scores(test_label, test_class_outputs, class_no)
    #         test_f1_, test_recall_, test_precision_, eTP, eTN, eFP, eFN, eP, eN = f1_score(test_label, test_class_outputs, class_no)
    #         test_iou += test_mean_iu_
    #         test_f1 += test_f1_
    #         #
    #         if (test_class_outputs == 1).sum() > 1 and (test_label == 1).sum() > 1:
    #             t_dist_ = hd95(test_class_outputs, test_label, class_no)
    #             test_h_dist += t_dist_
    #             test_h_dist_effective = test_h_dist_effective + 1
    #             #
    #         # save segmentation:
    #         save_name = save_path + '/test_' + str(ii) + '_seg.png'
    #         save_name_label = save_path + '/test_' + str(ii) + '_label.png'
    #         (b, c, h, w) = test_label.shape
    #         assert c == 1
    #         test_class_outputs = test_class_outputs.reshape(h, w).cpu().detach().numpy() > 0.5
    #         plt.imsave(save_name, test_class_outputs, cmap='gray')
    #         plt.imsave(save_name_label, test_label.reshape(h, w).cpu().detach().numpy(), cmap='gray')
    #         #
    #     test_iou = test_iou / (ii + 1)
    #     test_h_dist = test_h_dist / (test_h_dist_effective + 1)
    #     test_f1 = test_f1 / (ii + 1)
    # #
    # result_dictionary = {'Test IoU': str(test_iou),
    #                      'Test f1': str(test_f1),
    #                      'Test H-dist': str(test_h_dist)}
    #
    # ff_path = save_path + '/test_result_data.txt'
    # ff = open(ff_path, 'w')
    # ff.write(str(result_dictionary))
    # ff.close()
    # save model
    # ==========================
    # New testing
    # ==========================
    acc_total = []
    sensitivity_total = []
    specificity_total = []
    precision_total = []
    recall_total = []
    iou_total = []
    G_total = []
    mse_total = []
    auc_roc_total = []
    auc_pr_total = []
    F1_score_total = []

    epoch_threshold = num_epochs - 9

    for i in range(9):
        # model.load_state_dict(torch.load('./DRIVE_AV/checkpoints/20200824_all/AV_model_unet_13CP_epoch{}.pth'.format(511 + 10 * i)))
        current_model_name = dir_checkpoint + '/' + save_model_name + '_current_epoch' + str(epoch_threshold + 1 * i) + '.pth'
        print(current_model_name)
        # model.load_state_dict(torch.load(current_model_name))
        model = torch.load(current_model_name)
        # dir_checkpoint + '/' + save_model_name + '_' + str(epoch) + '.pth'
        model.eval()
        model.to(device=device)
        # test_score, acc, sensitivity, specificity, precision, G, F1_score_2 = eval_net_multitask(epoch, model, validateloader, device, mask=True, mode='vessel', model_name=model_name)
        accuracy_eva, iou_eva, precision_eva, recall_eva, f1_eva, specificity_eva, sensitivity_eva, g_eva, auc_pr_eva, auc_roc_eva, mse_eva = eval_net_multitask(epoch, model, testloader, device, mask=True, mode='vessel', model_name=model_name)

        acc_total.append(accuracy_eva)
        iou_total.append(iou_eva)
        precision_total.append(precision_eva)
        recall_total.append(recall_eva)
        F1_score_total.append(f1_eva)

        sensitivity_total.append(sensitivity_eva)
        specificity_total.append(specificity_eva)
        G_total.append(g_eva)
        auc_roc_total.append(auc_roc_eva)
        auc_pr_total.append(auc_pr_eva)
        mse_total.append(mse_eva)

    for n, batch in enumerate(testloader):
        #
        imgs, true_masks_main, true_masks_auxilary = batch['image'], batch['mask_main'], batch['mask_auxilary']
        imgs = imgs.to(device=device, dtype=torch.float32)
        # true_masks = true_masks_main.to(device=device, dtype=torch.float32)
        #
        b, c, h, w = imgs.size()
        #
        # print(b)
        # print(c)
        # print(h)
        # print(w)
        #
        if 'MTSARVSnet' in model_name:

            mask_pred, side_output1, side_output2, side_output3, output_v, side_output1_v, side_output2_v, side_output3_v = model(imgs)

        else:

            mask_pred = model(imgs)

        _, prediction = torch.max(mask_pred, dim=1)

        testoutput_original = np.asarray(prediction.cpu().detach().numpy(), dtype=np.uint8)
        testoutput_original = np.squeeze(testoutput_original, axis=0)
        testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)

        segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)

        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
        #
        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255

        prediction_name = 'seg_' + str(n) + '.png'
        full_error_map_name = os.path.join(save_path, prediction_name)
        imageio.imsave(full_error_map_name, segmentation_map)

        print(prediction_name + ' segmentation is saved.')

    # print('Accuracy (mean): ', np.mean(acc_total))
    # print('Sensitivity (mean): ', np.mean(sensitivity_total))
    # print('specificity (mean): ', np.mean(specificity_total))
    # print('precision (mean): ', np.mean(precision_total))
    # print('G (mean): ', np.mean(G_total))
    # print('F1_score_2 (mean): ', np.mean(F1_score_2_total))
    #
    # print('Accuracy (std): ', np.std(acc_total))
    # print('Sensitivity (std): ', np.std(sensitivity_total))
    # print('specificity (std): ', np.std(specificity_total))
    # print('precision (std): ', np.std(precision_total))
    # print('G (std): ', np.std(G_total))
    # print('F1_score_2 (std): ', np.std(F1_score_2_total))
    #
    # result_dictionary_mean = {'Test Accuracy mean': str(np.mean(acc_total)),
    #                           'Test Sensitivity mean': str(np.mean(sensitivity_total)),
    #                           'Test Specificity mean': str(np.mean(specificity_total)),
    #                           'Test Precision mean': str(np.mean(precision_total)),
    #                           'Test G mean': str(np.mean(G_total)),
    #                           'Test F1 mean': str(np.mean(F1_score_2))}
    #
    # result_dictionary_std = {'Test Accuracy std': str(np.std(acc_total)),
    #                           'Test Sensitivity std': str(np.std(sensitivity_total)),
    #                           'Test Specificity std': str(np.std(specificity_total)),
    #                           'Test Precision std': str(np.std(precision_total)),
    #                           'Test G std': str(np.std(G_total)),
    #                           'Test F1 std': str(np.std(F1_score_2))}

    print('Accuracy (mean): ', np.mean(acc_total))
    print('recall (mean): ', np.mean(recall_total))
    print('precision (mean): ', np.mean(precision_total))
    print('iou (mean): ', np.mean(iou_total))
    print('F1_score (mean): ', np.mean(F1_score_total))

    print('Accuracy (std): ', np.std(acc_total))
    print('recall (std): ', np.std(recall_total))
    print('precision (std): ', np.std(precision_total))
    print('iou (std): ', np.std(iou_total))
    print('F1_score (std): ', np.std(F1_score_total))

    # result_dictionary_mean = {'Test Accuracy mean': str(np.mean(acc_total)),
    #                           'Test Sensitivity mean': str(np.mean(sensitivity_total)),
    #                           'Test Specificity mean': str(np.mean(specificity_total)),
    #                           'Test Precision mean': str(np.mean(precision_total)),
    #                           'Test G mean': str(np.mean(G_total)),
    #                           'Test F1 mean': str(np.mean(F1_score_2))}
    #
    # result_dictionary_std = {'Test Accuracy std': str(np.std(acc_total)),
    #                           'Test Sensitivity std': str(np.std(sensitivity_total)),
    #                           'Test Specificity std': str(np.std(specificity_total)),
    #                           'Test Precision std': str(np.std(precision_total)),
    #                           'Test G std': str(np.std(G_total)),
    #                           'Test F1 std': str(np.std(F1_score_2))}

    result_dictionary_mean = {'Test Accuracy mean': str(np.mean(acc_total)),
                              'Test recall mean': str(np.mean(recall_total)),
                              'Test iou mean': str(np.mean(iou_total)),
                              'Test Precision mean': str(np.mean(precision_total)),
                              'Test F1 mean': str(np.mean(F1_score_total)),
                              'Test sensitivity mean': str(np.mean(sensitivity_total)),
                              'Test specifity mean': str(np.mean(specificity_total)),
                              'Test G mean': str(np.mean(G_total)),
                              'Test auc roc mean': str(np.mean(auc_roc_total)),
                              'Test auc pr mean': str(np.mean(auc_pr_total)),
                              'Test mse mean': str(np.mean(mse_total))}

    result_dictionary_std = {'Test Accuracy std': str(np.std(acc_total)),
                              'Test recall std': str(np.std(recall_total)),
                              'Test iou std': str(np.std(iou_total)),
                              'Test Precision std': str(np.std(precision_total)),
                              'Test F1 std': str(np.std(F1_score_total)),
                             'Test sensitivity std': str(np.std(sensitivity_total)),
                             'Test specifity std': str(np.std(specificity_total)),
                             'Test G std': str(np.std(G_total)),
                             'Test auc roc std': str(np.std(auc_roc_total)),
                             'Test auc pr std': str(np.std(auc_pr_total)),
                             'Test mse std': str(np.std(mse_total))
                             }

    save_path = saved_information_path + '/quantitative_results'

    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    ff_path_mean = save_path + '/test_result_mean.txt'
    ff_mean = open(ff_path_mean, 'w')
    ff_mean.write(str(result_dictionary_mean))
    ff_mean.close()

    ff_path_std = save_path + '/test_result_std.txt'
    ff_std = open(ff_path_std, 'w')
    ff_std.write(str(result_dictionary_std))
    ff_std.close()
    # ===========================
    stop = timeit.default_timer()
    #
    print('Time: ', stop - start)
    #
    save_model_name_full = saved_model_path + '/' + save_model_name + '_Final.pt'
    #
    path_model = save_model_name_full
    #
    torch.save(model, path_model)
    #
    print('\nTraining finished and final model saved\n')
    #
    # print('Test IoU: ' + str(test_iou) + '\n')
    # print('Test H-dist: ' + str(test_h_dist) + '\n')
    #

    return model

