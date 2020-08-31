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

    trainModels(data_directory='./data',
                dataset_name='DRIVE_AV',
                input_dim=3,
                class_no=4,
                repeat=1,
                train_batchsize=1,
                num_epochs=200,
                learning_rate=1e-3,
                image_size=(592, 592),
                multi_task=True,
                network='MTSARVSnet',
                log_tag='Retinal_20200824',
                save_threshold_epoch=190,
                save_interval_epoch=1)

    # network = 'MTSARVSnet',