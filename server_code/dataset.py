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
    def __init__(self, imgs_dir, masks_dir,  module_dir, img_size, dataset_name, transforms = train_transformer, train_or=True, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.module_dir = module_dir
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.transform = transforms
        self.train_or = train_or
        
        i = 0
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        self.ids_1 = [splitext(file)[0] for file in listdir(masks_dir)
                    if not file.startswith('.')]

        self.ids_2 = [splitext(file)[0] for file in listdir(module_dir)
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
    def crop_imgs(self, imgs, mask_array, module_array, img_size, k):

        img_h,img_w=imgs.shape[0], imgs.shape[1]
        target_h,target_w=4*img_size[0],4*img_size[1]  
        y_pos = np.random.randint(297,1323, size=1)
        x_pos = np.random.randint(297,1147, size=1)

        if len(imgs.shape)==3:
            d=imgs.shape[2]
            padded_final_img=np.zeros((592,592,d))
            padded_final_label=np.zeros((592,592,d))
            padded_final_module=np.zeros((592,592))
        elif len(imgs.shape)==2:
            padded_final_img=np.zeros((592,592))
            padded_final_module=np.zeros((592,592))


        for i in range(1,4):
            for j in range(1,4):
                #padded_final[((i-1)*3+j-1)*imgs.shape[0]:(((i-1)*3+j)*imgs.shape[0])]=imgs[:,(i-1)*280+200:(i-1)*280+792,(j-1)*390+100:(j-1)*390+692,...]
                '''
                padded_final_img[((i-1)*2+j-1)*imgs.shape[0]:(((i-1)*2+j)*imgs.shape[0]),...]=imgs[(i-1)*592+200:(i)*592+200,(j-1)*592+200:(j)*592+200,...]
                padded_final_label[((i-1)*2+j-1)*imgs.shape[0]:(((i-1)*2+j)*imgs.shape[0]),...]=mask_array[(i-1)*592+200:(i)*592+200,(j-1)*592+200:(j)*592+200,...]
                padded_final_module[((i-1)*2+j-1)*imgs.shape[0]:(((i-1)*2+j)*imgs.shape[0]),...]=module_array[(i-1)*592+200:(i)*592+200,(j-1)*592+200:(j)*592+200,...]

                final_img = Image.fromarray((padded_final_img[((i-1)*2+j-1)*imgs.shape[0]:(((i-1)*2+j)*imgs.shape[0]),...]).astype(np.uint8))
                final_label = Image.fromarray((padded_final_label[((i-1)*2+j-1)*imgs.shape[0]:(((i-1)*2+j)*imgs.shape[0]),...]).astype(np.uint8))
                final_module = Image.fromarray((padded_final_module[((i-1)*2+j-1)*imgs.shape[0]:(((i-1)*2+j)*imgs.shape[0]),...]).astype(np.uint8))
                '''
                print('xxxxxxxxxxxxxx', (i-1)*592+50-(i-1)*230)
                print('xxxxxxxxxxxxxx', (i)*592+50-(i-1)*230)
                print('yyyyyyyyyy', (j-1)*592+100-(j-1)*150)
                print('yyyyyyyyyy', (j)*592+100-(j-1)*150)

                padded_final_img=imgs[(i-1)*592+50-(i-1)*230:(i)*592+50-(i-1)*230,(j-1)*592+100-(j-1)*150:(j)*592+100-(j-1)*150,...]
                padded_final_label=mask_array[(i-1)*592+50-(i-1)*230:(i)*592+50-(i-1)*230,(j-1)*592+100-(j-1)*150:(j)*592+100-(j-1)*150,...]
                padded_final_module=module_array[(i-1)*592+50-(i-1)*230:(i)*592+50-(i-1)*230,(j-1)*592+100-(j-1)*150:(j)*592+100-(j-1)*150]

                #print(np.unique(padded_final_module))

                final_img = Image.fromarray((padded_final_img).astype(np.uint8))
                final_label = Image.fromarray((padded_final_label*255).astype(np.uint8))
                final_module = Image.fromarray((padded_final_module*255).astype(np.uint8))


                final_img.save('./data/LES-AV-patch/test/images/image_{}_{}_{}.png'.format(k,i,j))
                final_label.save('./data/LES-AV-patch/test/1st_manual/label_{}_{}_{}.png'.format(k,i,j))
                final_module.save('./data/LES-AV-patch/test/mask/mask_{}_{}_{}.gif'.format(k,i,j))

        
        #padded_final_img=imgs[(x_pos[0]-296):(x_pos[0]+296),y_pos[0]-296:y_pos[0]+296,...]
        #padded_final_label=mask_array[(x_pos[0]-296):(x_pos[0]+296),y_pos[0]-296:y_pos[0]+296,...]
        #padded_final_module=module_array[(x_pos[0]-296):(x_pos[0]+296),y_pos[0]-296:y_pos[0]+296,...]
  
        return padded_final_img, padded_final_label, padded_final_module

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def preprocess(self, pil_img, mask, module, dataset_name, img_size, train_or, k):
        #w, h = pil_img.size
        newW, newH = img_size[0], img_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))

        img_array = np.array(pil_img)
        mask_array = np.array(mask)/255
        #module_array = np.array(module)/255
        module_array = np.array(module)

        if dataset_name=='DRIVE_AV':
            img_array = self.pad_imgs(img_array, img_size)
            mask_array = self.pad_imgs(mask_array, img_size)
            module_array = self.pad_imgs(module_array, img_size)
            #print('@@@@@@@@@@@@@@', np.shape(img_array))
            #print('@@@@@@@@@@@@@@', np.shape(mask_array))
        
        if dataset_name=='LES-AV':
            img_array, mask_array, module_array = self.crop_imgs(img_array, mask_array, module_array, img_size, k)
            #mask_array = self.crop_imgs(mask_array, img_size)
            #module_array = self.crop_imgs(module_array, img_size)

        if train_or:
            if np.random.random()>0.5:
                img_array=img_array[:,::-1,:]    # flipped imgs
                mask_array=mask_array[:,::-1]
                module_array=module_array[:,::-1]

            angle = 3 * np.random.randint(120)
            img_array = rotate(img_array, angle, axes=(0, 1), reshape=False)
            #print('@@@@@@@@@@@@@@', np.shape(img_array))
            #print('@@@@@@@@@@@@@@', np.shape(mask_array))

            img_array = self.random_perturbation(img_array)
            mask_array = np.round(rotate(mask_array, angle, axes=(0, 1), reshape=False))
            module_array = np.round(rotate(module_array, angle, axes=(0, 1), reshape=False))


        mean=np.mean(img_array[img_array[...,0] > 00.0],axis=0)
        std=np.std(img_array[img_array[...,0] > 00.0],axis=0)

        #assert len(mean)==3 and len(std)==3
        #img_array=(img_array-mean)/std
        if dataset_name=='DRIVE_AV':
            img_array=(img_array-1.0*mean)/1.0*std
        if dataset_name=='LES-AV':
            img_array=(img_array-1.0*mean)/1.0*std
        if dataset_name=='LES-AV-patch':
            img_array=(img_array-1.0*mean)/1.0*std
            
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        if len(mask_array.shape) == 2:
            mask_array = np.expand_dims(mask_array, axis=2)

        #print(np.shape(img_array))

        image_array_img = Image.fromarray((img_array).astype(np.uint8))

        image_array_img.save('./aug_results/inside_img_{:02}.png'.format(k))
        #mask_array_img_squ = np.squeeze(mask_array)
        #mask_array_img_squ = Image.fromarray((mask_array_img_squ*255).astype(np.uint8))
        #image_array_img.save('./aug_results/new/inside_img_{:02}.png'.format(k))
        #mask_array_img_squ.save('./aug_results/new/inside_mask_{:02}.png'.format(k))
        if dataset_name=='DRIVE_AV':
            img_array = img_array.transpose((2, 0, 1))
            mask_array = mask_array.transpose((2, 0, 1))

            mask_array = np.where(mask_array > 0.5, 1, 0)

        if dataset_name=='LES-AV':
            img_array = img_array.transpose((2, 0, 1))
            mask_array = mask_array.transpose((2, 0, 1))

            mask_array = np.where(mask_array > 0.5, 1, 0)

        if dataset_name=='LES-AV-patch':
            img_array = img_array.transpose((2, 0, 1))
            mask_array = mask_array.transpose((2, 0, 1))

            mask_array = np.where(mask_array > 0.5, 1, 0)

        #print('!!!!!!!!!!!!!!', np.shape(img_array))
        #print('!!!!!!!!!!!!!!', np.shape(mask_array))

        k += 1

        return img_array, mask_array, module_array


    def __getitem__(self, i):
        idx = self.ids[i]
        idx_1 = self.ids_1[i]
        idx_2 = self.ids_2[i]
        #mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        mask_file = glob(self.masks_dir + idx_1  + '.*')
        #logging.info(f'Creating dataset with {len(mask_file)} mask')
        img_file = glob(self.imgs_dir + idx + '.*')
        module_file = glob(self.module_dir + idx_2 + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        module = Image.open(module_file[0])
        
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        

        if self.transform:
            img, mask, module = self.preprocess(img, mask, module, self.dataset_name, self.img_size, self.train_or, i)



        #save_image(torch.from_numpy(img), './aug_results/new/img_{:02}.png'.format(i))
        #save_image(torch.from_numpy(mask), './aug_results/new/mask_{:02}.png'.format(i))

        i += 1
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'module':torch.from_numpy(module).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')



