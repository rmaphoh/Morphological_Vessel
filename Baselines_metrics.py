import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import _ni_support

import torch.nn.functional as F

from PIL import Image

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

from eval import pad_imgs, dice_coeff, AUC_PR, AUC_ROC, threshold_by_otsu, pixel_values_in_mask, misc_measures
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, precision_recall_curve, balanced_accuracy_score, accuracy_score
from scipy import ndimage
# good references:
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/utilities/util_common.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/evaluation/pairwise_measures.py
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
#-----------------------------------------------


def intersectionAndUnion(imLab, imPred, numClass):

    # imPred, imLab = preprocessing_accuracy(imLab, imPred, numClass)

    #print(np.unique(imPred))
    #print(np.unique(imLab))

    # imPred = np.asarray(imPred).copy()
    # imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    accuracy = accuracy_score(imLab.flatten(), imPred.flatten())

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    iou_metric = area_intersection / area_union

    return iou_metric.mean(), accuracy


def f1_score(label_gt, label_pred, n_class):
    # threhold = torch.Tensor([0])

    # label_pred, label_gt = preprocessing_accuracy(label_gt, label_pred, n_class)

    assert len(label_gt) == len(label_pred)

    # precision = np.zeros(n_class, dtype=np.float32)
    # recall = np.zeros(n_class, dtype=np.float32)

    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()

    # precision[:] = precision_score(img_A, img_B, average='macro', labels=range(n_class))
    # recall[:] = recall_score(img_A, img_B, average='macro', labels=range(n_class))
    #
    precision = precision_score(img_A, img_B, average='macro')
    recall = recall_score(img_A, img_B, average='macro')
    #
    f1_metric = 2 * (recall * precision) / (recall + precision + 1e-8)
    #
    return f1_metric.mean(), recall.mean(), precision.mean()

# def iou_metric(outputs, labels):
#     # ========================================================================================================
#     # method 2
#     # adopted from CSAILVision:
#     zeros_outputs = torch.zeros_like(outputs)
#     ones_outputs = torch.ones_like(outputs)
#     outputs = torch.where((outputs > 0.5), ones_outputs, zeros_outputs)
#     # outputs = (outputs > 0.5)
#     imPred = np.asarray(outputs.cpu().detach()).copy()
#     imLab = np.asarray(labels.cpu().detach()).copy()
#     imPred += 1
#     imLab += 1
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     imPred = imPred * (imLab > 0)
#     # Compute area intersection:
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(
#         intersection, bins=2, range=(1, 2))
#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=2, range=(1, 2))
#     (area_lab, _) = np.histogram(imLab, bins=2, range=(1, 2))
#     area_union = area_pred + area_lab - area_intersection
#     iou = area_intersection / (area_union+1e-8)
#     return iou.mean()
# ==============================================================================================================
# accuracy:


def preprocessing_accuracy(label_true, label_pred, n_class):
    if n_class == 2:
        output_zeros = torch.zeros_like(label_pred)
        output_ones = torch.ones_like(label_pred)
        label_pred = torch.where((label_pred > 0.5), output_ones, output_zeros)
    label_pred = label_pred.cpu().detach()
    label_true = label_true.cpu().detach()
    label_pred = np.asarray(label_pred, dtype='int32')
    label_true = np.asarray(label_true, dtype='int32')

    return label_pred, label_true
#
# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# # https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# # https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
#


# def _fast_hist(label_true, label_pred, n_class):
#     label_pred, label_true = preprocessing_accuracy(label_true, label_pred, n_class)
#     mask = (label_true >= 0) & (label_true < n_class)
#     hist = np.bincount(
#         n_class * label_true[mask].astype(int) +
#         label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
#     return hist
# #
# #
# def segmentation_scores(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#       - fwavacc
#     """
#     # label_preds, label_trues = preprocessing_accuracy(label_trues, label_preds, n_class)
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     # iou:
#     iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
#     # iu = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) + 1e-8)
#     mean_iou = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     # iflat = label_preds.view(-1)
#     # tflat = label_trues.view(-1)
#     # intersection = (iflat * tflat).sum()
#     # union = iflat.sum() + tflat.sum()
#     #
#     # dice_score = (2. * intersection) / union
#     #
#     return fwavacc, acc_cls
# ==================================================================================


# def f1_score(label_gt, label_pred, n_class):
#     # threhold = torch.Tensor([0])
#     label_pred, label_gt = preprocessing_accuracy(label_gt, label_pred, n_class)
#     #
#     if len(label_gt.shape) == 4:
#         b, c, h, w = label_gt.shape
#         size = b * c
#     elif len(label_gt.shape) == 3:
#         c, h, w = label_gt.shape
#         size = c
#     #
#     assert len(label_gt) == len(label_pred)
#     #
#     precision = np.zeros(n_class, dtype=np.float32)
#     recall = np.zeros(n_class, dtype=np.float32)
#     img_A = np.array(label_gt, dtype=np.float32).flatten()
#     img_B = np.array(label_pred, dtype=np.float32).flatten()
#     precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
#     recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))
#     f1_metric = 2 * (recall * precision) / (recall + precision + 1e-10)
#     # For binary:
#     #
#     # CM = confusion_matrix(img_A, img_B)
#     # #
#     # TN = CM[0][0]
#     # FN = CM[1][0]
#     # TP = CM[1][1]
#     # FP = CM[0][1]
#     #
#     TN, FP, FN, TP = confusion_matrix(img_A, img_B, labels=[0.0, 1.0]).ravel()
#     #
#     # TP = np.sum(label_gt[label_gt == 1.0] == label_pred[label_pred == 1.0])
#     # TN = np.sum(label_gt[label_gt == 0.0] == label_pred[label_pred == 0.0])
#     # FP = np.sum(label_gt[label_gt == 1.0] == label_pred[label_pred == 0.0])
#     # FN = np.sum(label_gt[label_gt == 0.0] == label_pred[label_pred == 1.0])
#     # For multi-class:
#     # FP = CM.sum(axis=0) - np.diag(CM)
#     # FN = CM.sum(axis=1) - np.diag(CM)
#     # TP = np.diag(CM)
#     # TN = CM.sum() - (FP + FN + TP)
#     # FPs_Ns = (FP + 1e-8) / ((img_A == float(0.0)).sum() + 1e-8)
#     # FNs_Ps = (FN + 1e-8) / ((img_A == float(1.0)).sum() + 1e-8)
#     #
#     N = TN + FP
#     P = TP + FN
#     #
#     # FPs_Ns = (FP + 1e-10) / (Negatives + 1e-10)
#     # FNs_Ps = (FN + 1e-10) / (Positives + 1e-10)
#     # CM = np.zeros((2, 2), dtype=np.float32)
#     #
#     return f1_metric.mean(), recall.mean(), precision.mean(), TP / size, TN / size, FP / size, FN / size, P / size, N / size

##==========================================================================================

# Hausdorff distance metric:
# adopted from niftynet:
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/utilities/util_common.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/evaluation/pairwise_measures.py


# class MorphologyOps(object):
#     """
#     Class that performs the morphological operations needed to get notably
#     connected component. To be used in the evaluation
#     """
#     #
#     # I did NOT use this function because I don't want to post-processing interfere the segmentation evaluation.
#     #
#     def __init__(self, binary_img, neigh):
#         # assert len(binary_img.shape) == 3, 'currently supports 3d inputs only'
#         #
#         print(len(binary_img.shape))
#         if len(binary_img.shape) == 1:
#             #
#             binary_map_ = torch.cat((binary_img, binary_img, binary_img), dim=1)
#         #
#         self.binary_map = np.asarray(binary_map_, dtype=np.int8)
#         self.neigh = neigh
#
#     def border_map(self):
#         """
#         Creates the border for a 3D image
#         :return:
#         """
#         west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
#         east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
#         north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
#         south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
#         top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
#         bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
#         cumulative = west + east + north + south + top + bottom
#         border = ((cumulative < 6) * self.binary_map) == 1
#         return border
#
#     def foreground_component(self):
#         return ndimage.label(self.binary_map)


# def border_distance(label, output):
#     """
#     This functions determines the map of distance from the borders of the
#     segmentation and the reference and the border maps themselves
#
#     :return: distance_border_ref, distance_border_seg, border_ref,
#         border_seg
#     """
#     # border_ref = MorphologyOps(label, 4).border_map()
#     # border_seg = MorphologyOps(output, 4).border_map()
#     #
#     border_ref = label.cpu()
#     border_ref = np.asarray(border_ref, dtype=np.int8)
#     border_seg = output.cpu()
#     border_seg = np.asarray(border_seg, dtype=np.int8)
#     #
#     oppose_ref = 1 - label
#     oppose_seg = 1 - output
#     # euclidean distance transform
#     distance_ref = ndimage.distance_transform_edt(oppose_ref)
#     distance_seg = ndimage.distance_transform_edt(oppose_seg)
#     distance_border_seg = border_ref * distance_seg
#     distance_border_ref = border_seg * distance_ref
#     return distance_border_ref, distance_border_seg, border_ref, border_seg
#
#
# def getHausdorff(label, output):
#     """
#     This functions calculates the average symmetric distance and the
#     hausdorff distance between a segmentation and a reference image
#     :return: hausdorff distance and average symmetric distance
#     """
#     # label = label.squeeze(0)
#     # output = output.squeeze(1)
#     # label = label.cpu().detach()
#     # output = output.cpu().detach()
#     ref_border_dist, seg_border_dist, ref_border, seg_border = border_distance(label, output)
#     # average_distance = (np.sum(ref_border_dist) + np.sum(
#     #     seg_border_dist)) / (np.sum(label + output))
#     hausdorff_distance = np.max([np.max(ref_border_dist), np.max(seg_border_dist)])
#     seg_values = ref_border_dist[seg_border > 0]
#     ref_values = seg_border_dist[ref_border > 0]
#     if seg_values.size == 0 or ref_values.size == 0:
#         hausdorff95_distance = np.nan
#     else:
#         hausdorff95_distance = np.max([np.percentile(seg_values, 95), np.percentile(ref_values, 95)])
#
#     return hausdorff95_distance, hausdorff_distance


# ==================================

# def perf_measure(y_actual, y_hat):
#
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#
#     for i in range(len(y_hat)):
#         if y_actual[i]==y_hat[i]==1:
#            TP += 1
#         if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
#            FP += 1
#         if y_actual[i]==y_hat[i]==0:
#            TN += 1
#         if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
#            FN += 1
#
#     return TP, FP, TN, FN


# =============================
# reference :http://loli.github.io/medpy/_modules/medpy/metric/binary.html


def __surface_distances(result, reference, class_no, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result, reference = preprocessing_accuracy(reference, result, class_no)
    # reference = reference.cpu().detach().numpy()
    # result = result.cpu().detach().numpy()
    #
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hd95(result, reference, class_no, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, class_no, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, class_no, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    #
    hd95_mean = np.nanmean(hd95)
    return hd95_mean


# def eval_net_multitask(epoch, net, loader, device, mask, model_name):
#     """Evaluation without the densecrf with the dice coefficient"""
#     net.eval()
#     # mask_type = torch.float32 if net.n_classes == 1 else torch.long
#     n_val = len(loader)  # the number of batch
#     tot = 0
#     img_size = (592, 592)
#     module = Image.open('./data/DRIVE_AV/test/mask/01_test_mask.gif')
#     module = np.asarray(module) / 255
#
#     module_pad = pad_imgs(module, img_size).flatten()
#
#     # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#     for batch in loader:
#
#         if 'MTSARVSnet' in model_name:
#
#             imgs, true_masks_main, true_masks_auxilary = batch['image'], batch['mask_main'], batch['mask_auxilary']
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             true_masks = true_masks_main.to(device=device, dtype=torch.float32)
#
#         else:
#
#             imgs, true_masks = batch['image'], batch['mask']
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             true_masks = true_masks.to(device=device, dtype=torch.float32)
#
#         with torch.no_grad():
#
#             if 'MTSARVSnet' in model_name:
#
#                 mask_pred, side_output1, side_output2, side_output3, output_v, side_output1_v, side_output2_v, side_output3_v = net(imgs)
#
#             else:
#
#                 mask_pred = net(imgs)
#
#         # if net.output_dim > 1:
#         # tot += F.cross_entropy(mask_pred, true_masks).item()
#         # else:
#             # pred = torch.sigmoid(mask_pred)
#         pred = (mask_pred > 0.5).float()
#
#         tot += dice_coeff(pred, true_masks).item()
#
#         if mask:
#             # mask_pred_sigmoid = torch.sigmoid(mask_pred)
#             mask_pred_sigmoid_cpu = mask_pred.detach().cpu().numpy().flatten()
#             true_masks_cpu = true_masks.detach().cpu().numpy().flatten()
#
#             vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_masks_cpu, mask_pred_sigmoid_cpu, module_pad)
#             auc_roc = AUC_ROC(vessels_in_mask, generated_in_mask)
#             auc_pr = AUC_PR(vessels_in_mask, generated_in_mask)
#
#             binarys_in_mask = threshold_by_otsu(generated_in_mask)
#
#             ########################################
#             acc, sensitivity, specificity, precision, G, F1_score_2 = misc_measures(vessels_in_mask, binarys_in_mask)
#
#             ######################################
#             # print test images
#             '''
#             segmented_vessel=utils.remain_in_mask(generated, test_masks)
#             for index in range(segmented_vessel.shape[0]):
#                 Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_out_dir,str(n_round)+"_{:02}_segmented.png".format(index+1)))
#             '''
#             # pbar.update()
#
#     net.train()
#
#     return tot / n_val, acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr


def eval_net_multitask(epoch, net, loader, device, mask, mode, model_name):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # net_a.eval()
    # net_v.eval()
    ##################sigmoid or softmax
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    # mask_type = torch.float32 if net.n_classes == 1 else torch.float32
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    img_size = (592, 592)
    module = Image.open('./data/DRIVE_AV/test/mask/01_test_mask.gif')
    module = np.asarray(module) / 255

    # module_pad = pad_imgs(module, img_size).flatten()
    module_pad_1 = pad_imgs(module, img_size)

    module_pad_2 = np.expand_dims(module_pad_1, axis=0)
    module_pad = np.concatenate((module_pad_2, module_pad_2, module_pad_2), axis=0)

    precision_eva = 0
    recall_eva = 0
    accuracy_eva = 0
    iou_eva = 0
    f1_eva = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for n, batch in enumerate(loader):

        # imgs, true_masks = batch['image'], batch['mask']
        # imgs = imgs.to(device=device, dtype=torch.float32)
        # true_masks = true_masks.to(device=device, dtype=mask_type)

        # if 'MTSARVSnet' in model_name:
        #
        #     imgs, true_masks_main, true_masks_auxilary = batch['image'], batch['mask_main'], batch['mask_auxilary']
        #     imgs = imgs.to(device=device, dtype=torch.float32)
        #     true_masks = true_masks_main.to(device=device, dtype=torch.float32)
        #
        # else:

        imgs, true_masks_main, true_masks_auxilary = batch['image'], batch['mask_main'], batch['mask_auxilary']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks_main.to(device=device, dtype=torch.float32)

        # masks_pred_G = net_a(imgs)
        # masks_pred_G_sigmoid_A = torch.sigmoid(masks_pred_G)
        #
        # masks_pred_G = net_v(imgs)
        # masks_pred_G_sigmoid_V = torch.sigmoid(masks_pred_G)
        #
        # masks_pred_G_sigmoid_A_part = masks_pred_G_sigmoid_A.detach()
        # masks_pred_G_sigmoid_V_part = masks_pred_G_sigmoid_V.detach()

        # mask_pred, _, _, _ = net(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)

        if 'MTSARVSnet' in model_name:

            mask_pred, side_output1, side_output2, side_output3, output_v, side_output1_v, side_output2_v, side_output3_v = net(imgs)

        else:

            mask_pred = net(imgs)

        # mask_pred_artery = mask_pred[:, 0, :, :]
        # mask_pred_uncer = mask_pred[:, 1, :, :]
        # mask_pred_vein = mask_pred[:, 2, :, :]
        #
        # true_masks_artery = true_masks[:, 0, :, :]
        # true_masks_uncer = true_masks[:, 1, :, :]
        # true_masks_vein = true_masks[:, 2, :, :]

        # print(mask_pred.size())

        _, prediction = torch.max(mask_pred, dim=1)
        prediction = prediction.float()

        if n == 0:

            prediction_unique, prediction_counts = np.unique(prediction.cpu().detach().numpy(), return_counts=True)
            mask_unique, mask_counts = np.unique(true_masks.cpu().detach().numpy(), return_counts=True)

            print(np.asarray((prediction_unique, prediction_counts)).T)
            print(np.asarray((mask_unique, mask_counts)).T)

        prediction = prediction.cpu().detach()
        prediction = np.asarray(prediction, dtype='uint8').squeeze()

        true_masks = true_masks.cpu().detach()
        true_masks = np.asarray(true_masks, dtype='uint8').squeeze()

        # print(np.shape(prediction))
        # print(np.shape(true_masks))

        h, w = np.shape(prediction)

        # print(h)

        # artery:
        prediction_artery = np.zeros([h, w], dtype=np.int8)
        mask_artery = np.zeros([h, w], dtype=np.int8)
        prediction_artery[prediction == 1] = 1
        mask_artery[true_masks == 1] = 1
        # vein
        prediction_vein = np.zeros([h, w], dtype=np.int8)
        mask_vein = np.zeros([h, w], dtype=np.int8)
        prediction_vein[prediction == 2] = 1
        mask_vein[true_masks == 2] = 1

        # print(np.unique(prediction_artery))
        # print(np.unique(mask_artery))

        f1_artery, recall_artery, precision_artery = f1_score(label_gt=mask_artery, label_pred=prediction_artery, n_class=2)
        iou_artery, acc_artery = intersectionAndUnion(imLab=mask_artery, imPred=prediction_artery, numClass=2)

        f1_vein, recall_vein, precision_vein = f1_score(label_gt=mask_vein, label_pred=prediction_vein, n_class=2)
        iou_vein, acc_vein = intersectionAndUnion(imLab=mask_vein, imPred=prediction_vein, numClass=2)

        f1_ = (f1_artery + f1_vein) / 2
        recall_ = (recall_artery + recall_vein) / 2
        precision_ = (precision_artery + precision_vein) / 2
        iou_ = (iou_artery + iou_vein) / 2
        acc_ = (acc_artery + acc_vein) / 2

        f1_eva += f1_
        recall_eva += recall_
        precision_eva += precision_
        iou_eva += iou_
        accuracy_eva += acc_

            ##################sigmoid or softmax
        '''
        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        '''
        # if net.n_classes == 0:
        #     tot += F.cross_entropy(mask_pred, true_masks).item()
        # else:
        # pred = torch.sigmoid(mask_pred)
        
        # pred = (mask_pred > 0.5).float()
        # tot += dice_coeff(prediction, true_masks).item()

        # if mask:
        #
        #     if mode == 'vessel':
        #         ######################################
        #
        #         # based on the vessel
        #
        #         ########################################
        #
        #         # mask_pred_sigmoid = torch.sigmoid(mask_pred)
        #
        #         mask_pred_sigmoid = mask_pred
        #
        #         mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
        #         mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)
        #
        #         true_masks_cpu = true_masks.detach().cpu().numpy()
        #         true_masks_cpu = np.squeeze(true_masks_cpu)
        #
        #         true_masks_cpu = true_masks_cpu.transpose((1, 2, 0))
        #         mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))
        #
        #         binarys_in_mask_vessel = threshold_by_otsu(mask_pred_sigmoid_cpu)
        #
        #         # binarys_in_mask_vessel = (mask_pred_sigmoid_cpu > 0.5).float()
        #
        #         # encoded_pred = np.zeros(binarys_in_mask_vessel.shape[1:2], dtype=int)
        #         encoded_pred = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
        #         # print(np.shape(encoded_pred))
        #         # print(np.shape(true_masks_cpu))
        #         # print(np.unique(true_masks_cpu))
        #         encoded_gt = np.zeros(true_masks_cpu.shape[0:2], dtype=int)
        #
        #         # convert white pixels to green pixels (which are ignored)
        #         white_ind = np.where(np.logical_and(true_masks_cpu[:, :, 0] == 1, true_masks_cpu[:, :, 1] == 1, true_masks_cpu[:, :, 2] == 1))
        #         # print('white, ',np.shape(white_ind))
        #
        #         # print(type(white_ind))
        #
        #         if white_ind[0].size != 0:
        #             # print(np.shape(true_masks_cpu))
        #
        #             # true_masks_cpu[:,white_ind[0],white_ind[1]] = [0,1,0]
        #             true_masks_cpu[white_ind] = [0, 1, 0]
        #             # true_masks_cpu = [0,1,0]
        #
        #         white_ind_pre = np.where(np.logical_and(binarys_in_mask_vessel[:, :, 0] == 1, binarys_in_mask_vessel[:, :, 1] == 1, binarys_in_mask_vessel[:, :, 2] == 1))
        #         if white_ind_pre[0].size != 0:
        #             binarys_in_mask_vessel[white_ind_pre] = [0, 1, 0]
        #
        #         # translate the images to arrays suited for sklearn metrics
        #         arteriole = np.where(np.logical_and(true_masks_cpu[:, :, 0] == 1, true_masks_cpu[:, :, 1] == 0));
        #         encoded_gt[arteriole] = 1
        #         venule = np.where(np.logical_and(true_masks_cpu[:, :, 2] == 1, true_masks_cpu[:, :, 1] == 0));
        #         encoded_gt[venule] = 2
        #         # uncertainty = np.where(np.logical_and(true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:, 0] == 0, true_masks_cpu[:,:, 2] == 0)); encoded_gt[uncertainty] = 3
        #         arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:, :, 0] == 1, binarys_in_mask_vessel[:, :, 1] == 0));
        #         encoded_pred[arteriole] = 1
        #         venule = np.where(np.logical_and(binarys_in_mask_vessel[:, :, 2] == 1, binarys_in_mask_vessel[:, :, 1] == 0));
        #         encoded_pred[venule] = 2
        #         # uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred[uncertainty] = 3
        #         vessel_point = np.where(np.logical_and(encoded_gt[:, :] > 0, encoded_pred[:, :] > 0))
        #
        #         encoded_pred_vessel_point = encoded_pred[vessel_point].flatten()
        #         # print('encoded_pred_vessel_point range is:',np.unique(encoded_pred_vessel_point))
        #         # print('encoded_pred_vessel_point shape is:',np.shape(encoded_pred_vessel_point))
        #         encoded_gt_vessel_point = encoded_gt[vessel_point].flatten()
        #
        #         acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)

            # pbar.update()

    net.train()

    return accuracy_eva/(n+1), iou_eva/(n+1), precision_eva/(n+1), recall_eva/(n+1), f1_eva/(n+1)

    # if mode == 'vessel':
    #
    #     return tot / n_val, acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve
    #
    # else:
    #     return tot / n_val, acc, sensitivity, specificity, precision, G, F1_score, auc_roc, auc_pr