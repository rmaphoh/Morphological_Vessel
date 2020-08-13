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
from torchvision.utils import save_image
from scipy.ndimage import rotate
from PIL import Image, ImageEnhance

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    #transforms.Resize(256),
    #transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.Pad((13,4,14,4), fill=0, padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(360),
    #transforms.ColorJitter(brightness=[0.8,1.2], contrast=[0.8,1.2], saturation=[0.8,1.2], hue=[0.8,1.2]),
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



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,  img_size, transforms = train_transformer, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.transform = transforms
        i = 0
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
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def preprocess(self, pil_img, mask, img_size,k):
        #w, h = pil_img.size
        newW, newH = img_size[0], img_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))

        img_array = np.array(pil_img)
        mask_array = np.array(mask)/255

        img_array = self.pad_imgs(img_array, img_size)
        mask_array = self.pad_imgs(mask_array, img_size)
        #print('@@@@@@@@@@@@@@', np.shape(img_array))
        #print('@@@@@@@@@@@@@@', np.shape(mask_array))

        if np.random.random()>0.5:
            img_array=img_array[:,::-1,:]    # flipped imgs
            mask_array=mask_array[:,::-1]

        angle = 3 * np.random.randint(120)
        img_array = rotate(img_array, angle, axes=(0, 1), reshape=False)
        #print('@@@@@@@@@@@@@@', np.shape(img_array))
        #print('@@@@@@@@@@@@@@', np.shape(mask_array))
        img_array = self.random_perturbation(img_array)
        mask_array = np.round(rotate(mask_array, angle, axes=(0, 1), reshape=False))

        mean_r=np.mean(img_array[...,0][img_array[...,0] > 00.0],axis=0)
        std_r=np.std(img_array[...,0][img_array[...,0] > 00.0],axis=0)

        mean_g=np.mean(img_array[...,1][img_array[...,0] > 00.0],axis=0)
        std_g=np.std(img_array[...,1][img_array[...,0] > 00.0],axis=0)

        mean_b=np.mean(img_array[...,2][img_array[...,0] > 00.0],axis=0)
        std_b=np.std(img_array[...,2][img_array[...,0] > 00.0],axis=0)
        #print('!!!!!!!!!!!', len(mean))
        #print('!!!!!!!!!!!', len(std))

        #assert len(mean)==3 and len(std)==3
        #img_array=(img_array-mean)/std
        img_array[...,0]=(img_array[...,0]-mean_r)/std_r
        img_array[...,1]=(img_array[...,1]-mean_g)/std_g
        img_array[...,2]=(img_array[...,2]-mean_b)/std_b
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        if len(mask_array.shape) == 2:
            mask_array = np.expand_dims(mask_array, axis=2)

        #print(np.shape(img_array))

        #image_array_img = Image.fromarray((img_array*255).astype(np.uint8))
        #image_array_img.save('./aug_results/new/inside_img_{:02}.png'.format(k))
        #mask_array_img_squ = np.squeeze(mask_array)
        #mask_array_img_squ = Image.fromarray((mask_array_img_squ*255).astype(np.uint8))
        #image_array_img.save('./aug_results/new/inside_img_{:02}.png'.format(k))
        #mask_array_img_squ.save('./aug_results/new/inside_mask_{:02}.png'.format(k))
        img_array = img_array.transpose((2, 0, 1))
        mask_array = mask_array.transpose((2, 0, 1))
        #print('!!!!!!!!!!!!!!', np.shape(img_array))
        #print('!!!!!!!!!!!!!!', np.shape(mask_array))

        k += 1

        return img_array, mask_array


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
        

        if self.transform:
            img, mask = self.preprocess(img, mask, self.img_size, i)



        #save_image(torch.from_numpy(img), './aug_results/new/img_{:02}.png'.format(i))
        #save_image(torch.from_numpy(mask), './aug_results/new/mask_{:02}.png'.format(i))

        i += 1
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')



