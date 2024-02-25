import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop

from Config import Config
from model import AesFA_test
from blocks import test_model_load

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def do_transform(img, osize):
    transform = Compose([Resize(size=osize),
                        CenterCrop(size=osize),
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform(img).unsqueeze(0)

def save_img(config, cont_name, sty_h_name, sty_l_name, content, style_h, style_l, stylized):
    real_A = im_convert(content)
    real_B_1 = im_convert(style_h)
    real_B_2 = im_convert(style_l)
    trs_AtoB = im_convert(stylized)

    A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
    B1_image = Image.fromarray((real_B_1[0] * 255.0).astype(np.uint8))
    B2_image = Image.fromarray((real_B_2[0] * 255.0).astype(np.uint8))
    trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

    cont_name = cont_name.split('/')[-1].split('.')[0]
    sty_h_name = sty_h_name.split('/')[-1].split('.')[0]
    sty_l_name = sty_l_name.split('/')[-1].split('.')[0]

    A_image.save('{}/{:s}_content.jpg'.format(config.img_dir, cont_name))
    B1_image.save('{}/{:s}_high_style.jpg'.format(config.img_dir, sty_h_name))
    B2_image.save('{}/{:s}_low_style.jpg'.format(config.img_dir, sty_l_name))
    trs_image.save('{}/stylized_{:s}_{:s}_{:s}.jpg'.format(config.img_dir, cont_name, sty_h_name, sty_l_name))

def main():
    config = Config()
    if not os.path.exists(config.img_dir):
        os.makedirs(config.img_dir)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Version:', config.file_n)
    print(device)
    
    with torch.no_grad():
        ## Model load
        model = AesFA_test(config)

        ## Load saved model
        ckpt = config.ckpt_dir + '/main.pth'
        print("checkpoint: ", ckpt)
        model = test_model_load(checkpoint=ckpt, model=model)
        model.to(device)

        ## Style Blending
        real_A = Image.open(config.content_img).convert('RGB')
        style_high = Image.open(config.style_high_img).convert('RGB')
        style_low = Image.open(config.style_low_img).convert('RGB')

        real_A = do_transform(real_A, config.blend_load_size).to(device)
        style_high = do_transform(style_high, config.blend_load_size).to(device)
        style_low = do_transform(style_low, config.blend_load_size).to(device)

        stylized, during = model.style_blending(real_A, style_high, style_low)
        save_img(config, config.content_img, config.style_high_img, config.style_low_img, real_A, style_high, style_low, stylized)
        print("Time:", during)

if __name__ == '__main__':
    main()