import os
import sys
import logging
import errno
import torch
import timeit

import argparse

import tqdm

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

from tensorboardX import SummaryWriter
# ====================================

from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

# ====================================

from Baselines_models import MTSARVSnet
from Baselines_metrics import segmentation_scores, f1_score


def trainModels(dataset_tag,
                dataset_name,
                data_directory,
                input_dim,
                class_no,
                repeat,
                train_batchsize,
                augmentation,
                num_epochs,
                learning_rate,
                network,
                log_tag,
                main_loss='dice'):
    #
    for j in range(1, repeat + 1):
        #
        repeat_str = str(j)
        #
        if network == 'MTSARVSnet':
            #
            Exp = MTSARVSnet(input_dim=input_dim, output_dim=class_no)
            #
            Exp_name = 'MTSARVSnet_' + \
                       '_train_batch_' + str(train_batchsize) + \
                       '_repeat_' + str(repeat_str) + \
                       '_main_loss_' + main_loss + \
                       '_lr_' + str(learning_rate) + \
                       '_epoch_' + str(num_epochs) + dataset_name

        # ====================================================================================================================================================================
        trainloader, validateloader = getData(data_directory, dataset_name, dataset_tag, train_batchsize, augmentation)
        # ===================
        trainSingleModel(Exp,
                         Exp_name,
                         num_epochs,
                         learning_rate,
                         dataset_name,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         losstag=main_loss,
                         class_no=class_no,
                         log_tag=log_tag)


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, data_augment):

    # img_out_dir="./{}/AV_segmentation_results_{}_{}".format(args.dataset, args.dis, args.gan2seg)
    # dir_checkpoint="./{}/checkpoints/AV_model_{}_{}".format(args.dataset,args.dis,args.gan2seg)
    # auc_out_dir="{}/AV_auc_{}_{}".format(args.dataset, args.dis, args.gan2seg)
    train_dir= "./data/DRIVE_AV/training/images/"
    dir_mask = "./data/DRIVE_AV/training/2st_manual/"

    # # create files
    # if not os.path.isdir(img_out_dir):
    #     os.makedirs(img_out_dir)
    # if not os.path.isdir(dir_checkpoint):
    #     os.makedirs(dir_checkpoint)
    # if not os.path.isdir(auc_out_dir):
    #     os.makedirs(auc_out_dir)

    val_percent = 0.2

    dataset = BasicDataset(train_dir, dir_mask, (592, 592))
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    return train_loader, val_loader

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     datasettag,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     losstag,
                     log_tag,
                     class_no):

    #
    device = torch.device('cuda')
    #
    save_model_name = model_name
    #
    saved_information_path = '../../Results'
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
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

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
        train_recall = 0
        train_precision = 0
        train_effective_h = 0

        # for j, (images, labels) in enumerate(trainloader):
        for j, batch_current in enumerate(trainloader):
            #
            # print(np.unique(labels))
            images = batch_current['image']
            labels = batch_current['mask']

            optimizer.zero_grad()
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            if 'MTSARVSnet' in model_name:

                output, side_output1, side_output2, side_output3 = model(images)

                # print(output.size())

                _, prob_outputs = torch.max(output, dim=1)
                _, prob_outputs_side1 = torch.max(side_output1, dim=1)
                _, prob_outputs_side2 = torch.max(side_output2, dim=1)
                _, prob_outputs_side3 = torch.max(side_output3, dim=1)

                sideloss1 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output1, dim=1), labels.squeeze(1))
                sideloss2 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output2, dim=1), labels.squeeze(1))
                sideloss3 = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(side_output3, dim=1), labels.squeeze(1))

                mainloss = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(output, dim=1), labels.squeeze(1))

                loss = mainloss + (sideloss1 + sideloss2 + sideloss3) / 3

            loss.backward()
            optimizer.step()

        class_outputs = prob_outputs

        # only calculate training accuracy when labels are available in training data:
        if torch.max(labels) == 1.0 and torch.min(labels) == 0.0:

            # if (class_outputs == 1).sum() > 1 and (labels == 1).sum() > 1:
            #     #
            #     dist_ = hd95(class_outputs, labels, class_no)
            #     train_h_dists += dist_
            #     train_effective_h = train_effective_h + 1
                #
            train_mean_iu_ = segmentation_scores(labels, class_outputs, class_no)
            #
            train_f1_, train_recall_, train_precision_, TPs_, TNs_, FPs_, FNs_, Ps_, Ns_ = f1_score(labels, class_outputs, class_no)
            #
            train_main_loss += loss.item()
            train_f1 += train_f1_
            train_iou += train_mean_iu_
            train_recall += train_recall_
            train_precision += train_precision_

        # Evaluate at the end of each epoch:
        model.eval()
        with torch.no_grad():
            #
            validate_iou = 0
            validate_f1 = 0
            validate_h_dist = 0
            validate_h_dist_effective = 0
            #
            # for i, (val_img, val_label, val_name) in enumerate(validateloader):
            for i, val_batch_current in enumerate(validateloader):

                val_img = val_batch_current['image']
                val_label = val_batch_current['mask']

                val_img = val_img.to(device=device, dtype=torch.float32)
                val_label = val_label.to(device=device, dtype=torch.long)

                assert torch.max(val_label) == 1.0
                assert torch.min(val_label) == 0.0

                if 'MTSARVSnet' in model_name:

                    val_outputs, val_outputs_side1, val_outputs_side2, val_outputs_side3 = model(val_img)
                    _, val_class_outputs = torch.max(val_outputs, dim=1)

                eval_mean_iu_ = segmentation_scores(val_label, val_class_outputs, class_no)
                eval_f1_, eval_recall_, eval_precision_, eTP, eTN, eFP, eFN, eP, eN = f1_score(val_label, val_class_outputs, class_no)
                validate_iou += eval_mean_iu_
                validate_f1 += eval_f1_
                #
                # if (val_class_outputs == 1).sum() > 1 and (val_label == 1).sum() > 1:
                #     v_dist_ = hd95(val_class_outputs, val_label, class_no)
                #     validate_h_dist += v_dist_
                #     validate_h_dist_effective = validate_h_dist_effective + 1

        print(
            'Step [{}/{}], '
            'Train main loss: {:.4f}, '
            'Train iou: {:.4f}, '
            'val iou:{:.4f}, '.format(epoch + 1, num_epochs,
                                      train_main_loss / (j + 1),
                                      train_iou / (j + 1),
                                      validate_iou / (i + 1)))
        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #
        writer.add_scalars('acc metrics', {'train iou': train_iou / (j+1),
                                           'train hausdorff dist': train_h_dists / (train_effective_h+1),
                                           'val iou': validate_iou / (i + 1),
                                           'val f1': validate_f1 / (i + 1)}, epoch + 1)

        writer.add_scalars('loss values', {'main loss': train_main_loss / (j+1)}, epoch + 1)

        for param_group in optimizer.param_groups:
            #
            if epoch == (num_epochs - 10):
                param_group['lr'] = learning_rate * 0.1
                # param_group['lr'] = learning_rate * ((1 - epoch / num_epochs) ** 0.999)

    # # ===============
    # Testing:
    # =================
    # save_path = saved_information_path + '/Visual_results'
    #
    # try:
    #
    #     os.mkdir(save_path)
    #
    # except OSError as exc:
    #
    #     if exc.errno != errno.EEXIST:
    #
    #         raise
    #
    #     pass
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
    print('\nTraining finished and model saved\n')
    #
    # print('Test IoU: ' + str(test_iou) + '\n')
    # print('Test H-dist: ' + str(test_h_dist) + '\n')
    #
    return model

