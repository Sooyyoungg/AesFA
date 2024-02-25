from path import Path
import glob
# import torch
import torch.nn as nn
# import pandas as pd
# import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomCrop
import random

Image.MAX_IMAGE_PIXELS = 1000000000

class DataSplit(nn.Module):
    def __init__(self, config, phase='train'):
        super(DataSplit, self).__init__()

        self.transform = Compose([Resize(size=[config.load_size, config.load_size]),
                                RandomCrop(size=(config.crop_size, config.crop_size)),
                                ToTensor(),
                                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        if phase == 'train':
            # Content image data
            img_dir = Path(config.content_dir+'/train')
            self.images = self.get_data(img_dir)
            if config.data_num < len(self.images):
                self.images = random.sample(self.images, config.data_num)

            # Style image data
            sty_dir = Path(config.style_dir+'/train')
            self.style_images = self.get_data(sty_dir)
            if len(self.images) < len(self.style_images):
                self.style_images = random.sample(self.style_images, len(self.images))
            elif len(self.images) > len(self.style_images):
                ratio = len(self.images) // len(self.style_images)
                bias = len(self.images) - ratio * len(self.style_images)
                self.style_images = self.style_images * ratio
                self.style_images += random.sample(self.style_images, bias)
            assert len(self.images) == len(self.style_images)
            
        elif phase == 'test':
            img_dir = Path(config.content_dir)
            self.images = self.get_data(img_dir)[:config.data_num]
            
            sty_dir = Path(config.style_dir)
            self.style_images = self.get_data(sty_dir)[:config.data_num]
        
        print('content dir:', img_dir)
        print('style dir:', sty_dir)
            
    def __len__(self):
        return len(self.images)
    
    def get_data(self, img_dir):
        file_type = ['*.jpg', '*.png', '*.jpeg', '*.tif']
        imgs = []
        for ft in file_type:
            imgs += sorted(img_dir.glob(ft))
        images = sorted(imgs)
        return images

    def __getitem__(self, index):
        cont_img = self.images[index]
        cont_img = Image.open(cont_img).convert('RGB')
        cont_img = self.transform(cont_img)

        sty_img = self.style_images[index]
        sty_img = Image.open(sty_img).convert('RGB')
        sty_img = self.transform(sty_img)

        return {'content_img': cont_img, 'style_img': sty_img}