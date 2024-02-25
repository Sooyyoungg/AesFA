import os
import torch
import numpy as np
import thop
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop

from Config import Config
from DataSplit import DataSplit
from model import AesFA_test
from blocks import test_model_load


def load_img(img_name, img_size, device):
    img = Image.open(img_name).convert('RGB')
    img = do_transform(img, img_size).to(device)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # make batch dimension
    return img

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def do_transform(img, osize):
    transform = Compose([Resize(size=osize),  # Resize to keep aspect ratio
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

    A_image.save('{}/{:s}_content_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    B_image.save('{}/{:s}_style_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    trs_image.save('{}/{:s}_stylized_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    
    if freq:
        trs_AtoB_high = im_convert(high)
        trs_AtoB_low = im_convert(low)

        trsh_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
        trsl_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))
        
        trsh_image.save('{}/{:s}_stylizing_high_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
        trsl_image.save('{}/{:s}_stylizing_low_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))

        
def main():
    config = Config()

    config.gpu = 0
    device = torch.device('cuda:'+str(config.gpu) if torch.cuda.is_available() else 'cpu')
    print('Version:', config.file_n)
    print(device)
    
    with torch.no_grad():
        ## Data Loader
        test_bs = 1
        test_data = DataSplit(config=config, phase='test')
        data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=test_bs, shuffle=False, num_workers=16, pin_memory=False)
        print("Test: ", test_data.__len__(), "images: ", len(data_loader_test), "x", test_bs, "(batch size) =", test_data.__len__())

        ## Model load
        ckpt = config.ckpt_dir + '/main.pth'
        print("checkpoint: ", ckpt)
        model = AesFA_test(config)
        model = test_model_load(checkpoint=ckpt, model=model)
        model.to(device)

        if not os.path.exists(config.img_dir):
            os.makedirs(config.img_dir)

        ## Start Testing
        freq = False                # whether save high, low frequency images or not
        count = 0
        t_during = 0

        contents = test_data.images
        styles = test_data.style_images
        if config.multi_to_multi:   # one content image, N style image
            tot_imgs = len(contents) * len(styles)
            for idx in range(len(contents)):
                cont_name = contents[idx]           # path of content image
                content = load_img(cont_name, config.test_content_size, device)

                for i in range(len(styles)):
                    sty_name = styles[i]            # path of style image
                    style = load_img(sty_name, config.test_style_size, device)
                    
                    if freq:
                        stylized, stylized_high, stylized_low, during = model(content, style, freq)
                        save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                    else:
                        stylized, during = model(content, style, freq)
                        save_img(config, cont_name, sty_name, content, style, stylized)

                    count += 1
                    print(count, idx+1, i+1, during)
                    t_during += during
                    flops, params = thop.profile(model, inputs=(content, style, freq))
                    print("GFLOPS: %.4f, Params: %.4f"% (flops/1e9, params/1e6))
                    print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=config.gpu) / 1024. / 1024. / 1024.))

        else:
            tot_imgs = len(contents)
            for idx in range(len(contents)):
                cont_name = contents[idx]
                content = load_img(cont_name, config.test_content_size, device)

                sty_name = styles[idx]
                style = load_img(sty_name, config.test_style_size, device)

                if freq:
                    stylized, stylized_high, stylized_low, during = model(content, style, freq)
                    save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                else:
                    stylized, during = model(content, style, freq)
                    save_img(config, cont_name, sty_name, content, style, stylized)

                t_during += during
                flops, params = thop.profile(model, inputs=(content, style, freq))
                print("GFLOPS: %.4f, Params: %.4f" % (flops / 1e9, params / 1e6))
                print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=config.gpu) / 1024. / 1024. / 1024.))


        t_during = float(t_during / (len(contents) * len(styles)))
        print("[AesFA] Content size:", config.test_content_size, "Style size:", config.test_style_size,
              " Total images:", tot_imgs, "Avg Testing time:", t_during)

            
if __name__ == '__main__':
    main()
