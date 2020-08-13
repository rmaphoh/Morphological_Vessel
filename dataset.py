from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms, utils
import random

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

'''
mean=np.mean(fundus_imgs[index,...][fundus_imgs[index,...,0] > 40.0],axis=0)
std=np.std(fundus_imgs[index,...][fundus_imgs[index,...,0] > 40.0],axis=0)
assert len(mean)==3 and len(std)==3
fundus_imgs[index,...]=(fundus_imgs[index,...]-mean)/std
'''

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,  img_size, transforms = train_transformer, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.transform = transforms

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {(self.ids)} ')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pad_imgs(self, imgs, img_size):
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

    @classmethod
    def preprocess(self, pil_img, img_size):
        #w, h = pil_img.size
        newW, newH = img_size[0], img_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)
        
        img_size_target = img_size

        #img_nd = self.pad_imgs(img_nd, img_size_target) 

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        return img_nd


    def __getitem__(self, i):
        idx = self.ids[i]
        #mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        mask_file = glob(self.masks_dir + idx  + '.*')
        #logging.info(f'Creating dataset with {len(mask_file)} mask')
    
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        


        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.cuda.manual_seed(seed) 

        if self.transform:
            img = self.transform(img)
            
        random.seed(seed) # apply this seed to target tranfsorms
        torch.cuda.manual_seed(seed) # needed for torchvision 0.7

        if self.transform:
            mask = self.transform(mask)

        img = self.preprocess(img, self.img_size)
        mask = self.preprocess(mask, self.img_size)

        #print('the range of img is: ',np.unique(img))
        #print('the range of mask is: ',np.unique(mask))
        '''
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        '''
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')