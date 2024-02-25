import os
import torch
import numpy as np
from PIL import Image
import glob
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop

from Config import Config
from DataSplit import DataSplit
from model import AesFA_test
from blocks import test_model_load


def load_img(img_name, img_size, device):
    img = Image.open(img_name).convert('RGB')
    img = do_transform(img, img_size).to(device)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return img

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def do_transform(img, osize):
    # if config.phase == 'test':
    #     osize = config.test_load_size
    # elif config.phase == 'style_blending':
    #     osize = config.blend_load_size
    transform = Compose([Resize(size=osize),
                        CenterCrop(size=osize),
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform(img)

def save_img(config, cont_name, sty_name, content, style, stylized, freq=False, high=None, low=None):
    real_A = im_convert(content)
    real_B = im_convert(style)
    trs_AtoB = im_convert(stylized)
    
    A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
    B_image = Image.fromarray((real_B[0] * 255.0).astype(np.uint8))
    trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

    if config.phase == 'test':
        A_image.save('{}/content/{:s}_content.jpg'.format(config.img_dir, cont_name.stem))
        B_image.save('{}/style/{:s}_style.jpg'.format(config.img_dir, sty_name.stem))
        trs_image.save('{}/stylized/{:s}_stylized_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    else:
        A_image.save('{}/content/{:s}_content.jpg'.format(config.img_dir, cont_name))
        B_image.save('{}/style/{:s}_style.jpg'.format(config.img_dir, sty_name))
        trs_image.save('{}/stylized/{:s}_stylized_{:s}.jpg'.format(config.img_dir, cont_name, sty_name))
    
    if freq:
        trs_AtoB_high = im_convert(high)
        trs_AtoB_low = im_convert(low)

        trsh_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
        trsl_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))
        
        trsh_image.save('{}/{:s}_stylizing_high_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
        trsl_image.save('{}/{:s}_stylizing_low_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))

        
def main():
    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Version:', config.file_n)
    print(device)
    
    with torch.no_grad():
        ## Model load
        ckpt = config.ckpt_dir + '/main.pth'
        print("checkpoint: ", ckpt)
        model = AesFA_test(config)
        model = test_model_load(checkpoint=ckpt, model=model)
        model.to(device)

        if not os.path.exists(config.img_dir):
            os.makedirs(config.img_dir)
            os.makedirs(config.img_dir+'/content')
            os.makedirs(config.img_dir+'/style')
            os.makedirs(config.img_dir+'/stylized')

        ## Start Testing
        count = 0
        t_during = 0
        if config.phase == 'test':
            ## Data Loader
            test_data = DataSplit(config=config, phase='test')
            contents = test_data.images
            styles = test_data.style_images
            print("# of contents:", len(contents))
            print("# of styles:", len(styles))
        
            for idx in range(len(contents)):
                cont_name = contents[idx]
                content = load_img(cont_name, config.test_content_size, device)

                for i in range(len(styles)):
                    sty_name = styles[i]
                    style = load_img(sty_name, config.test_style_size, device)

                    freq = False
                    if freq:
                        stylized, stylized_high, stylized_low, during = model(content, style, freq)
                        save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                    else:
                        stylized, during = model(content, style, freq)
                        save_img(config, cont_name, sty_name, content, style, stylized)

                    count += 1
                    print(count, idx+1, i+1, during)
                    t_during += during
                    
        elif config.phase == 'style_blending':
            contents = sorted(glob.glob(config.blend_dir+'/content/*.jpg'))
            for content in contents:
                cont_name = content.split('/')[-1].split('.')[0]
                content = load_img(cont_name, config.blend_load_size, device)
                
                style_h = config.style_high_img
                style_l = config.style_low_img

                sty_name = style_h.split('/')[-1].split('.')[0]
                style_h = Image.open(style_h).convert('RGB')
                style_h = do_transform(config, style_h).to(device)
                if len(style_h.shape) == 3:
                    style_h = style_h.unsqueeze(0)

                style_l = Image.open(style_l).convert('RGB')
                style_l = do_transform(config, style_l).to(device)
                if len(style_l.shape) == 3:
                    style_l = style_l.unsqueeze(0)

                stylized, during = model.style_blending(content, style_h, style_l)
                save_img(config, cont_name, sty_name, content, style_h, stylized)

        t_during = float(t_during / (len(contents) * len(styles)))
        print("[AesFA] Total images:", len(contents) * len(styles), "Avg Testing time:", t_during)

            
if __name__ == '__main__':
    main()
