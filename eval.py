import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Function
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, mean_squared_error
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



def pixel_values_in_mask(true_vessels, pred_vessels, module_pad, train_or):

    if train_or=='val':
        true_vessels = np.squeeze(true_vessels)
        pred_vessels = np.squeeze(pred_vessels)
        #print(np.shape(module_pad))
        #print(np.shape(true_vessels))
        true_vessels = (true_vessels[module_pad != 0])
        pred_vessels = (pred_vessels[module_pad != 0])

        assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
        assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0


    return true_vessels.flatten(), pred_vessels.flatten()
    #return true_vessels, pred_vessels

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
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)
    #print(np.shape(cm))
    #print('#################3')
    #print(len(np.shape(cm)))

    try:
        acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        G = np.sqrt(sensitivity*specificity)
        F1_score_2 = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou
    
    except:

        return 0,0,0,0,0,0,0,0

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

def eval_net(epoch, net, net_a, net_v, loader, device, mask, mode, train_or):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    ##################sigmoid or softmax
    #mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32 if net.n_classes == 1 else torch.float32
    n_val = len(loader)  # the number of batch
    tot_a=0
    sent_a=0
    spet_a=0
    pret_a=0
    G_t_a=0
    F1t_a=0
    mset_a=0
    iout_a=0
    auc_roct_a=0
    auc_prt_a=0

    tot_v=0
    sent_v=0
    spet_v=0
    pret_v=0
    G_t_v=0
    F1t_v=0
    mset_v=0
    iout_v=0
    auc_roct_v=0
    auc_prt_v=0

    tot_u=0
    sent_u=0
    spet_u=0
    pret_u=0
    G_t_u=0
    F1t_u=0
    mset_u=0
    iout_u=0
    auc_roct_u=0
    auc_prt_u=0

    tot=0
    sent=0
    spet=0
    pret=0
    G_t=0
    F1t=0
    mset=0
    iout=0
    auc_roct=0
    auc_prt=0

    img_size = (592,592)
    module = Image.open('./data/DRIVE_AV/test/mask/01_test_mask.gif')
    module = np.asarray(module)/255

    #module_pad = pad_imgs(module, img_size).flatten()
    module_pad_1 = pad_imgs(module, img_size)

    module_pad_2 = np.expand_dims(module_pad_1,axis=0)
    module_pad = np.concatenate((module_pad_2, module_pad_2, module_pad_2), axis=0)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, module_pad = batch['image'], batch['mask'], batch['module']
            
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
            
            if net.n_classes == 0:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            '''

            if mask:

                if mode== 'whole':
                    ########################################

                    # based on the whole images

                    ########################################

                    mask_pred_sigmoid = torch.sigmoid(mask_pred)
                    
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                    mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)

                    true_masks_cpu = true_masks.detach().cpu().numpy()
                    true_masks_cpu = np.squeeze(true_masks_cpu)

                    module_pad_cpu = module_pad.detach().cpu().numpy()
                    module_pad_cpu = np.squeeze(module_pad_cpu)

                    true_masks_cpu = true_masks_cpu.transpose((1, 2, 0))
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))

                    #binarys_in_mask_vessel=threshold_by_otsu(mask_pred_sigmoid_cpu)
                    binarys_in_mask_vessel=((mask_pred_sigmoid_cpu)>0.5).astype('float')

                    #encoded_pred = np.zeros(binarys_in_mask_vessel.shape[1:2], dtype=int)
                    encoded_pred_a = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
                    encoded_pred_u = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
                    encoded_pred_v = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
                    #print(np.shape(encoded_pred))
                    #print(np.shape(true_masks_cpu))
                    #print(np.unique(true_masks_cpu))
                    encoded_gt_a = np.zeros(true_masks_cpu.shape[0:2], dtype=int)
                    encoded_gt_u = np.zeros(true_masks_cpu.shape[0:2], dtype=int)
                    encoded_gt_v = np.zeros(true_masks_cpu.shape[0:2], dtype=int)

                    
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
                    arteriole = np.where(np.logical_and(true_masks_cpu[:,:,0] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt_a[arteriole] = 1
                    venule = np.where(np.logical_and(true_masks_cpu[:,:,2] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt_v[venule] = 1
                    uncertainty = np.where(np.logical_and(true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:, 0] == 0, true_masks_cpu[:,:, 2] == 0)); encoded_gt_u[uncertainty] = 1
                    arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred_a[arteriole] = 1
                    venule = np.where(np.logical_and(binarys_in_mask_vessel[:,:,2] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred_v[venule] = 1
                    uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred_u[uncertainty] = 1
                    #vessel_point = np.where(np.logical_and(encoded_gt[:,:]>0, encoded_pred[:,:] > 0))

                    count_artery = np.sum(encoded_gt_a==1)
                    count_vein = np.sum(encoded_gt_v==1)
                    count_uncertainty = np.sum(encoded_gt_u==1)
                    count_total = count_artery + count_vein + count_uncertainty


                    ###############################################
                    ##########################################
                    #artery
                    #######################################
                    encoded_gt_vessel_point_a, encoded_pred_vessel_point_a = pixel_values_in_mask(encoded_gt_a, encoded_pred_a, module_pad_cpu, train_or)

                    auc_roc_a=AUC_ROC(encoded_gt_vessel_point_a,encoded_pred_vessel_point_a)
                    auc_pr_a=AUC_PR(encoded_gt_vessel_point_a, encoded_pred_vessel_point_a)

                    #### ONly consider artery and vein to the background
                    acc_ve_a, sensitivity_ve_a, specificity_ve_a, precision_ve_a, G_ve_a, F1_score_ve_a, mse_a, iou_a = misc_measures(encoded_gt_vessel_point_a, encoded_pred_vessel_point_a)
            
                    tot_a+=acc_ve_a
                    sent_a+=sensitivity_ve_a
                    spet_a+=specificity_ve_a
                    pret_a+=precision_ve_a
                    G_t_a+=G_ve_a
                    F1t_a+=F1_score_ve_a
                    mset_a+=mse_a
                    iout_a+=iou_a
                    auc_roct_a+=auc_roc_a
                    auc_prt_a+=auc_pr_a

                    
                    ##########################################
                    #vein
                    #######################################
                    encoded_gt_vessel_point_v, encoded_pred_vessel_point_v = pixel_values_in_mask(encoded_gt_v, encoded_pred_v, module_pad_cpu, train_or)

                    auc_roc_v=AUC_ROC(encoded_gt_vessel_point_v,encoded_pred_vessel_point_v)
                    auc_pr_v=AUC_PR(encoded_gt_vessel_point_v, encoded_pred_vessel_point_v)

                    #### ONly consider artery and vein to the background
                    acc_ve_v, sensitivity_ve_v, specificity_ve_v, precision_ve_v, G_ve_v, F1_score_ve_v, mse_v, iou_v = misc_measures(encoded_gt_vessel_point_v, encoded_pred_vessel_point_v)
            
                    tot_v+=acc_ve_v
                    sent_v+=sensitivity_ve_v
                    spet_v+=specificity_ve_v
                    pret_v+=precision_ve_v
                    G_t_v+=G_ve_v
                    F1t_v+=F1_score_ve_v
                    mset_v+=mse_v
                    iout_v+=iou_v
                    auc_roct_v+=auc_roc_v
                    auc_prt_v+=auc_pr_v

                    
                    ##########################################
                    #uncertainty
                    #######################################
                    encoded_gt_vessel_point_u, encoded_pred_vessel_point_u = pixel_values_in_mask(encoded_gt_u, encoded_pred_u, module_pad_cpu,train_or )

                    auc_roc_u=AUC_ROC(encoded_gt_vessel_point_u,encoded_pred_vessel_point_u)
                    auc_pr_u=AUC_PR(encoded_gt_vessel_point_u, encoded_pred_vessel_point_u)

                    #### ONly consider artery and vein to the background
                    acc_ve_u, sensitivity_ve_u, specificity_ve_u, precision_ve_u, G_ve_u, F1_score_ve_u, mse_u, iou_u = misc_measures(encoded_gt_vessel_point_u, encoded_pred_vessel_point_u)
            
                    tot_u+=acc_ve_u
                    sent_u+=sensitivity_ve_u
                    spet_u+=specificity_ve_u
                    pret_u+=precision_ve_u
                    G_t_u+=G_ve_u
                    F1t_u+=F1_score_ve_u
                    mset_u+=mse_u
                    iout_u+=iou_u
                    auc_roct_u+=auc_roc_u
                    auc_prt_u+=auc_pr_u

                    tot+=(count_artery*acc_ve_a + count_vein*acc_ve_v + count_uncertainty*acc_ve_u)/count_total
                    sent+=(count_artery*sensitivity_ve_a + count_vein*sensitivity_ve_v + count_uncertainty*sensitivity_ve_u)/count_total
                    spet+=(count_artery*specificity_ve_a + count_vein*specificity_ve_v + count_uncertainty*specificity_ve_u)/count_total
                    pret+=(count_artery*precision_ve_a + count_vein*precision_ve_v + count_uncertainty*precision_ve_u)/count_total
                    G_t+=(count_artery*G_ve_a + count_vein*G_ve_v + count_uncertainty*G_ve_u)/count_total
                    F1t+=(count_artery*F1_score_ve_a + count_vein*F1_score_ve_v + count_uncertainty*F1_score_ve_u)/count_total
                    mset+=(count_artery*mse_a + count_vein*mse_v + count_uncertainty*mse_u)/count_total
                    iout+=(count_artery*iou_a + count_vein*iou_v + count_uncertainty*iou_u)/count_total
                    auc_roct+=(count_artery*auc_roc_a + count_vein*auc_roc_v + count_uncertainty*auc_roc_u)/count_total
                    auc_prt+=(count_artery*auc_pr_a + count_vein*auc_pr_v + count_uncertainty*auc_pr_u)/count_total


                if mode == 'artery':
                    ###################################### 

                    # based on the artery

                    ######################################## 

                    mask_pred_sigmoid = torch.sigmoid(mask_pred)
                    
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                    mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)

                    true_masks_cpu = true_masks.detach().cpu().numpy()
                    true_masks_cpu = np.squeeze(true_masks_cpu)

                    module_pad_cpu = module_pad.detach().cpu().numpy()
                    module_pad_cpu = np.squeeze(module_pad_cpu)

                    true_masks_cpu = true_masks_cpu.transpose((1, 2, 0))
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))

                    #binarys_in_mask_vessel=threshold_by_otsu(mask_pred_sigmoid_cpu)
                    binarys_in_mask_vessel=((mask_pred_sigmoid_cpu)>0.5).astype('float')

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
                    #venule = np.where(np.logical_and(true_masks_cpu[:,:,2] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[venule] = 2
                    #uncertainty = np.where(np.logical_and(true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:, 0] == 0, true_masks_cpu[:,:, 2] == 0)); encoded_gt[uncertainty] = 3
                    arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[arteriole] = 1
                    #venule = np.where(np.logical_and(binarys_in_mask_vessel[:,:,2] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[venule] = 2
                    #uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred[uncertainty] = 3
                    #vessel_point = np.where(np.logical_and(encoded_gt[:,:]>0, encoded_pred[:,:] > 0))

                    #module_pad = module_pad.transpose((1, 2, 0))
                    #print(np.shape(module_pad))
                    encoded_gt_vessel_point, encoded_pred_vessel_point = pixel_values_in_mask(encoded_gt, encoded_pred, module_pad_cpu,train_or )

                    auc_roc=AUC_ROC(encoded_gt_vessel_point,encoded_pred_vessel_point)
                    auc_pr=AUC_PR(encoded_gt_vessel_point, encoded_pred_vessel_point)
                    #encoded_pred_vessel_point = encoded_pred.flatten()
                    #print('encoded_pred_vessel_point range is:',np.unique(encoded_pred_vessel_point))
                    #print('encoded_pred_vessel_point shape is:',np.shape(encoded_pred_vessel_point))
                    #encoded_gt_vessel_point = encoded_gt.flatten()
                
                    #### ONly select the cross domain
                    #acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)

                    #### ONly consider artery and vein to the background
                    acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve, mse = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)
            
                    tot+=acc_ve
                    sent+=sensitivity_ve
                    spet+=specificity_ve
                    pret+=precision_ve
                    G_t+=G_ve
                    F1t+=F1_score_ve
                    mset+=mse
                    auc_roct+=auc_roc
                    auc_prt+=auc_pr

                if mode== 'vein':
                    ###################################### 

                    # based on the vein

                    ######################################## 


                    mask_pred_sigmoid = torch.sigmoid(mask_pred)
                    
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                    mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)

                    true_masks_cpu = true_masks.detach().cpu().numpy()
                    true_masks_cpu = np.squeeze(true_masks_cpu)

                    module_pad_cpu = module_pad.detach().cpu().numpy()
                    module_pad_cpu = np.squeeze(module_pad_cpu)

                    true_masks_cpu = true_masks_cpu.transpose((1, 2, 0))
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))

                    #binarys_in_mask_vessel=threshold_by_otsu(mask_pred_sigmoid_cpu)
                    binarys_in_mask_vessel=((mask_pred_sigmoid_cpu)>0.5).astype('float')

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
                    #arteriole = np.where(np.logical_and(true_masks_cpu[:,:,0] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[arteriole] = 1
                    venule = np.where(np.logical_and(true_masks_cpu[:,:,2] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[venule] = 1
                    #uncertainty = np.where(np.logical_and(true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:, 0] == 0, true_masks_cpu[:,:, 2] == 0)); encoded_gt[uncertainty] = 3
                    #arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[arteriole] = 1
                    venule = np.where(np.logical_and(binarys_in_mask_vessel[:,:,2] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[venule] = 1
                    #uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred[uncertainty] = 3
                    #vessel_point = np.where(np.logical_and(encoded_gt[:,:]>0, encoded_pred[:,:] > 0))

                    #module_pad = module_pad.transpose((1, 2, 0))
                    encoded_gt_vessel_point, encoded_pred_vessel_point = pixel_values_in_mask(encoded_gt, encoded_pred, module_pad_cpu,train_or )

                    auc_roc=AUC_ROC(encoded_gt_vessel_point,encoded_pred_vessel_point)
                    auc_pr=AUC_PR(encoded_gt_vessel_point, encoded_pred_vessel_point)
                    #encoded_pred_vessel_point = encoded_pred.flatten()
                    #print('encoded_pred_vessel_point range is:',np.unique(encoded_pred_vessel_point))
                    #print('encoded_pred_vessel_point shape is:',np.shape(encoded_pred_vessel_point))
                    #encoded_gt_vessel_point = encoded_gt.flatten()
                
                    #### ONly select the cross domain
                    #acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)

                    #### ONly consider artery and vein to the background
                    acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve, mse = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)
            
                    tot+=acc_ve
                    sent+=sensitivity_ve
                    spet+=specificity_ve
                    pret+=precision_ve
                    G_t+=G_ve
                    F1t+=F1_score_ve
                    mset+=mse
                    auc_roct+=auc_roc
                    auc_prt+=auc_pr

                if mode== 'uncertainty':
                    ###################################### 

                    # based on the uncertainty

                    ######################################## 

                    mask_pred_sigmoid = torch.sigmoid(mask_pred)
                    
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                    mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)

                    true_masks_cpu = true_masks.detach().cpu().numpy()
                    true_masks_cpu = np.squeeze(true_masks_cpu)
                    
                    module_pad_cpu = module_pad.detach().cpu().numpy()
                    module_pad_cpu = np.squeeze(module_pad_cpu)

                    true_masks_cpu = true_masks_cpu.transpose((1, 2, 0))
                    mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))

                    #binarys_in_mask_vessel=threshold_by_otsu(mask_pred_sigmoid_cpu)
                    binarys_in_mask_vessel=((mask_pred_sigmoid_cpu)>0.5).astype('float')

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
                    #arteriole = np.where(np.logical_and(true_masks_cpu[:,:,0] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[arteriole] = 1
                    #venule = np.where(np.logical_and(true_masks_cpu[:,:,2] == 1, true_masks_cpu[:,:,1] == 0)); encoded_gt[venule] = 1
                    uncertainty = np.where(np.logical_and(true_masks_cpu[:,:,1] == 1, true_masks_cpu[:,:, 0] == 0, true_masks_cpu[:,:, 2] == 0)); encoded_gt[uncertainty] = 1
                    #arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[arteriole] = 1
                    #venule = np.where(np.logical_and(binarys_in_mask_vessel[:,:,2] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred[venule] = 1
                    uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred[uncertainty] = 1
                    #vessel_point = np.where(np.logical_and(encoded_gt[:,:]>0, encoded_pred[:,:] > 0))

                    #module_pad = module_pad.transpose((1, 2, 0))
                    encoded_gt_vessel_point, encoded_pred_vessel_point = pixel_values_in_mask(encoded_gt, encoded_pred, module_pad_cpu,train_or )

                    auc_roc=AUC_ROC(encoded_gt_vessel_point,encoded_pred_vessel_point)
                    auc_pr=AUC_PR(encoded_gt_vessel_point, encoded_pred_vessel_point)
                    #encoded_pred_vessel_point = encoded_pred.flatten()
                    #print('encoded_pred_vessel_point range is:',np.unique(encoded_pred_vessel_point))
                    #print('encoded_pred_vessel_point shape is:',np.shape(encoded_pred_vessel_point))
                    #encoded_gt_vessel_point = encoded_gt.flatten()
                
                    #### ONly select the cross domain
                    #acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)

                    #### ONly consider artery and vein to the background
                    acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve, mse = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)
            
                    tot+=acc_ve
                    sent+=sensitivity_ve
                    spet+=specificity_ve
                    pret+=precision_ve
                    G_t+=G_ve
                    F1t+=F1_score_ve
                    mset+=mse
                    auc_roct+=auc_roc
                    auc_prt+=auc_pr

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
                
                    #### ONly select the cross domain
                    #acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)

                    #### ONly consider artery and vein to the background
                    acc_ve, sensitivity_ve, specificity_ve, precision_ve, G_ve, F1_score_ve = misc_measures(encoded_gt_vessel_point, encoded_pred_vessel_point)
            
                    tot+=acc_ve
                    sent+=sensitivity_ve
                    spet+=specificity_ve
                    pret+=precision_ve
                    G_t+=G_ve
                    F1t+=F1_score_ve

            pbar.update()

    net.train()
    
    if mode== 'vessel':
        return  tot / n_val, tot/ n_val, sent/ n_val, spet/ n_val, pret/ n_val, G_t/ n_val, F1t/ n_val
    
    else:
        return  tot / n_val, tot/ n_val, sent/ n_val, spet/ n_val, pret/ n_val, G_t/ n_val, F1t/ n_val, auc_roct/ n_val, auc_prt/ n_val, mset/ n_val, iout/ n_val, \
            tot_a / n_val, tot_a/ n_val, sent_a/ n_val, spet_a/ n_val, pret_a/ n_val, G_t_a/ n_val, F1t_a/ n_val, auc_roct_a/ n_val, auc_prt_a/ n_val, mset_a/ n_val, iout_a/ n_val, \
                tot_v / n_val, tot_v/ n_val, sent_v/ n_val, spet_v/ n_val, pret_v/ n_val, G_t_v/ n_val, F1t_v/ n_val, auc_roct_v/ n_val, auc_prt_v/ n_val, mset_v/ n_val, iout_v/ n_val, \
                    tot_u / n_val, tot_u/ n_val, sent_u/ n_val, spet_u/ n_val, pret_u/ n_val, G_t_u/ n_val, F1t_u/ n_val, auc_roct_u/ n_val, auc_prt_u/ n_val, mset_u/ n_val, iout_u/ n_val


