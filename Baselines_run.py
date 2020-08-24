import torch
# import sys
# sys.path.append("..")
# from NNTrain import trainModels
# from NNTrain_norm_ensemble import trainModels
# from Train import trainModels

from Baselines_train import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':

    # trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #             dataset_name='CARVE2014',
    #             dataset_tag='1slice_corners_ssl_176',
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=2,
    #             augmentation='flip',
    #             num_epochs=1,
    #             learning_rate=1e-3,
    #             width=16,
    #             network='Unet',
    #             log_tag='ERFNet_ssl',
    #             consistency_loss='mse',
    #             main_loss='dice',
    #             self_addition=True)

    trainModels(data_directory='/home/moucheng/projects_codes/Morphological_Vessel/',
                dataset_name='data',
                dataset_tag='DRIVE_AV',
                input_dim=3,
                class_no=3,
                repeat=1,
                train_batchsize=2,
                augmentation='flip',
                num_epochs=1,
                learning_rate=1e-3,
                network='MTSARVSnet',
                log_tag='Retinal',
                main_loss='dice')
