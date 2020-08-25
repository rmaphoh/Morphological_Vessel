import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Function
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import filters
import numpy as np
from PIL import Image


def pad_imgs( imgs, img_size):
    img_h,img_w=imgs.shape[0], imgs.shape[1]
    target_h,target_w=img_size[0],img_size[1] 
    if len(imgs.shape)==3:
        d=imgs.shape[2]
        padded=np.zeros((target_h, target_w,d))
    elif len(imgs.shape)==2:
        padded=np.zeros((target_h, target_w))
    padded[(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
    #print(np.shape(padded))
    return padded



def pixel_values_in_mask(true_vessels, pred_vessels, module_pad):


    true_vessels = np.squeeze(true_vessels)
    pred_vessels = np.squeeze(pred_vessels)
    #print(np.shape(module_pad))
    #print(np.shape(true_vessels))
    true_vessels = (true_vessels[module_pad != 0])
    pred_vessels = (pred_vessels[module_pad != 0])

    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0


    return true_vessels.flatten(), pred_vessels.flatten()

def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    #print(type(true_vessel_arr.flatten()))
    #print(np.shape(true_vessel_arr.flatten()))
    #fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)

    #AUC_ROC=roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    AUC_ROC=roc_auc_score(true_vessel_arr, pred_vessel_arr)
    return AUC_ROC

def threshold_by_otsu(pred_vessels):
    
    # cut by otsu threshold
    threshold=filters.threshold_otsu(pred_vessels)
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1

    return pred_vessels_bin

def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    #print(np.shape(cm))
    print('#################3')
    print(len(np.shape(cm)))

    if len(np.shape(cm)) !=2:
        return 0,0,0,0,0,0
    else:
        acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        G = np.sqrt(sensitivity*specificity)
        F1_score_2 = 2*precision*sensitivity/(precision+sensitivity)
        return acc, sensitivity, specificity, precision, G, F1_score_2

def print_metrics(itr, **kargs):
    print ("*** Round {}  ====> ".format(itr),)
    for name, value in kargs.items():
        print ( "{} : {}, ".format(name, value),)
    print ("")
    sys.stdout.flush()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001

        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def eval_net(epoch, net, net_a, net_v, loader, device, mask, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    net_a.eval()
    net_v.eval()
    ##################sigmoid or softmax
    #mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32 if net.n_classes == 1 else torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    img_size = (592,592)
    module = Image.open('./data/DRIVE_AV/test/mask/01_test_mask.gif')
    module = np.asarray(module)/255

    #module_pad = pad_imgs(module, img_size).flatten()
    module_pad_1 = pad_imgs(module, img_size)

    module_pad_2 = np.expand_dims(module_pad_1,axis=0)
    module_pad = np.concatenate((module_pad_2, module_pad_2, module_pad_2), axis=0)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                
                masks_pred_G = net_a(imgs)
                masks_pred_G_sigmoid_A = torch.sigmoid(masks_pred_G)
                
                masks_pred_G = net_v(imgs)
                masks_pred_G_sigmoid_V = torch.sigmoid(masks_pred_G)

                masks_pred_G_sigmoid_A_part = masks_pred_G_sigmoid_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_sigmoid_V.detach()

                mask_pred,_,_,_ = net(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)

                mask_pred_artery=mask_pred[:,0,:,:]
                mask_pred_uncer = mask_pred[:,1,:,:]
                mask_pred_vein=mask_pred[:,2,:,:]
                true_masks_artery=true_masks[:,0,:,:]
                true_masks_uncer=true_masks[:,1,:,:]
                true_masks_vein=true_masks[:,2,:,:] 
            
            ##################sigmoid or softmax
            '''
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            '''
            if net.n_classes == 0:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

            if mask:

                if mode== 'whole':
                    ########################################

                    # based on the whole images

                    ########################################

                    mask_pred_sigmoid = torch.sigmoid(mask_pred)
                    '''
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy().flatten()
                    true_masks_cpu = true_masks.detach().cpu().numpy().flatten()
                    '''
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                    true_masks_cpu = true_masks.detach().cpu().numpy()

                    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_masks_cpu, mask_pred_sigmoid_cpu, module_pad )


                    auc_roc=AUC_ROC(vessels_in_mask,generated_in_mask)
                    auc_pr=AUC_PR(vessels_in_mask, generated_in_mask)
                    
                    binarys_in_mask=threshold_by_otsu(generated_in_mask)
                    
                    
                    acc, sensitivity, specificity, precision, G, F1_score = misc_measures(vessels_in_mask, binarys_in_mask)

                if mode == 'artery':
                    ###################################### 

                    # based on the artery

                    ######################################## 

                    mask_pred_artery_sigmoid = torch.sigmoid(mask_pred_artery)

                    mask_pred_artery_sigmoid_cpu = mask_pred_artery_sigmoid.detach().cpu().numpy()
                    
                    true_masks_artery_cpu = true_masks_artery.detach().cpu().numpy()

                    vessels_in_mask_artery, generated_in_mask_artery = pixel_values_in_mask(true_masks_artery_cpu, mask_pred_artery_sigmoid_cpu, module_pad_1 )


                    auc_roc=AUC_ROC(vessels_in_mask_artery,generated_in_mask_artery)
                    auc_pr=AUC_PR(vessels_in_mask_artery, generated_in_mask_artery)
                    
                    binarys_in_mask_artery=threshold_by_otsu(generated_in_mask_artery)
                    
                    
                    acc, sensitivity, specificity, precision, G, F1_score = misc_measures(vessels_in_mask_artery, binarys_in_mask_artery)

                if mode== 'vein':
                    ###################################### 

                    # based on the vein

                    ######################################## 

                    mask_pred_vein_sigmoid = torch.sigmoid(mask_pred_vein)

                    mask_pred_vein_sigmoid_cpu = mask_pred_vein_sigmoid.detach().cpu().numpy()
                    
                    true_masks_vein_cpu = true_masks_vein.detach().cpu().numpy()

                    vessels_in_mask_vein, generated_in_mask_vein = pixel_values_in_mask(true_masks_vein_cpu, mask_pred_vein_sigmoid_cpu, module_pad_1 )


                    auc_roc=AUC_ROC(vessels_in_mask_vein,generated_in_mask_vein)
                    auc_pr=AUC_PR(vessels_in_mask_vein, generated_in_mask_vein)
                    
                    binarys_in_mask_vein=threshold_by_otsu(generated_in_mask_vein)
                    
                    
                    acc, sensitivity, specificity, precision, G, F1_score = misc_measures(vessels_in_mask_vein, binarys_in_mask_vein)

                if mode== 'uncertainty':
                    ###################################### 

                    # based on the uncertainty

                    ######################################## 

                    mask_pred_uncer_sigmoid = torch.sigmoid(mask_pred_uncer)

                    mask_pred_uncer_sigmoid_cpu = mask_pred_uncer_sigmoid.detach().cpu().numpy()
                    
                    true_masks_uncer_cpu = true_masks_uncer.detach().cpu().numpy()

                    vessels_in_mask_uncer, generated_in_mask_uncer = pixel_values_in_mask(true_masks_uncer_cpu, mask_pred_uncer_sigmoid_cpu, module_pad_1 )


                    auc_roc=AUC_ROC(vessels_in_mask_uncer,generated_in_mask_uncer)
                    auc_pr=AUC_PR(vessels_in_mask_uncer, generated_in_mask_uncer)
                    
                    binarys_in_mask_uncer=threshold_by_otsu(generated_in_mask_uncer)
                    
                    
                    acc, sensitivity, specificity, precision, G, F1_score = misc_measures(vessels_in_mask_uncer, binarys_in_mask_uncer)


                if mode== 'vessel':
                    ###################################### 

                    # based on the vessel

                    ######################################## 
                    

                    mask_pred_sigmoid = torch.sigmoid(mask_pred)
                    
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                    mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)

                    true_masks_cpu = true_masks.detach().cpu().numpy()
                    true_masks_cpu = np.squeeze(true_masks_cpu)

                    true_masks_cpu = true_masks_cpu.transpose((1, 2, 0))
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))

                    binarys_in_mask_vessel=threshold_by_otsu(mask_pred_sigmoid_cpu)

                    #encoded_pred = np.zeros(binarys_in_mask_vessel.shape[1:2], dtype=int)
                    encoded_pred = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
                    #print(np.shape(encoded_pred))
                    #print(np.shape(true_masks_cpu))
                    #print(np.unique(true_masks_cpu))
                    encoded_gt = np.zeros(true_masks_cpu.shape[0:2], dtype=int)
                    
                    
                    # convert white pixels to green pixels (which are ignored)
                    white_ind = np.where(np.logical_and(true_masks_cpu[:,:,0] == 1, true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:,2] == 1))
                    #print('white, ',np.shape(white_ind))

                    #print(type(white_ind))

                    if white_ind[0].size != 0:
                        #print(np.shape(true_masks_cpu))

                        #true_masks_cpu[:,white_ind[0],white_ind[1]] = [0,1,0]
                        true_masks_cpu[white_ind] = [0,1,0]
                        #true_masks_cpu = [0,1,0]

                    white_ind_pre = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 1, binarys_in_mask_vessel[:,:,2] == 1))
                    if white_ind_pre[0].size != 0:
                        binarys_in_mask_vessel[white_ind_pre] = [0,1,0]
                    
                        
                    # translate the images to arrays suited for sklearn metrics
                    arteriole = np.where(np.logical_and(true_masks_cpu[:,:,0] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[arteriole] = 1
                    venule = np.where(np.logical_and(true_masks_cpu[:,:,2] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[venule] = 2
                    #uncertainty = np.where(np.logical_and(true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:, 0] == 0, true_masks_cpu[:,:, 2] == 0)); encoded_gt[uncertainty] = 3
                    arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[arteriole] = 1
                    venule = np.where(np.logical_and(binarys_in_mask_vessel[:,:,2] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[venule] = 2
                    #uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred[uncertainty] = 3
                    vessel_point = np.where(np.logical_and(encoded_gt[:,:]>0, encoded_pred[:,:] > 0))

                    encoded_pred_vessel_point = encoded_pred[vessel_point].flatten()
                    #print('encoded_pred_vessel_point range is:',np.unique(encoded_pred_vessel_point))
                    #print('encoded_pred_vessel_point shape is:',np.shape(encoded_pred_vessel_point))
                    encoded_gt_vessel_point = encoded_gt[vessel_point].flatten()
                

                    acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)
            
            pbar.update()

    net.train()
    
    if mode== 'vessel':
        return  tot / n_val, acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve
    
    else:
        return  tot / n_val, acc, sensitivity, specificity, precision, G, F1_score, auc_roc, auc_pr
