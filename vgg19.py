import torch
import torch.nn as nn
import numpy as np

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    
    nn.ReLU(),  # relu1-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    
    nn.ReLU(),  # relu2-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    
    nn.ReLU(),  # relu3-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    
    nn.ReLU(),  # relu4-1, this is the last layer used
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    
    nn.ReLU(),  # relu5-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class VGG_loss(nn.Module):
    def __init__(self, config, vgg):
        super(VGG_loss, self).__init__()
       
        self.config = config

        vgg_pretrained = config.vgg_model
        vgg.load_state_dict(torch.load(vgg_pretrained))
        vgg = nn.Sequential(*list(vgg.children())[:43])         # depends on what layers you want to load
        vgg_enc_layers = list(vgg.children())
        
        self.n_layers = 4
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:3])     # ~ conv1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[3:10])   # conv1_1 ~ conv2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[10:17])  # conv2_1 ~ conv3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[17:30])  # conv3_1 ~ conv4_1

        self.mse_loss = nn.MSELoss()

        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1
    def encode_with_vgg_intermediate(self, input):
        results = [input]
        for i in range(self.n_layers):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
   
    # extract relu3_1
    def encode_vgg_content(self, input):
        for i in range(3):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input
        
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def efdm_single(self, style, trans):
        B, C, W, H = style.size(0), style.size(1), style.size(2), style.size(3)
        
        value_style, index_style = torch.sort(style.view(B, C, -1))
        value_trans, index_trans = torch.sort(trans.view(B, C, -1))
        inverse_index = index_trans.argsort(-1)
        
        return self.mse_loss(trans.view(B, C,-1), value_style.gather(-1, inverse_index))

    def perceptual_loss(self, content, style, trs_img):
        # normalization for putting images as inputs to VGG
        content = content.permute(0, 2, 3, 1)
        style = style.permute(0, 2, 3, 1)
        trs_img = trs_img.permute(0, 2, 3, 1)
        
        content = content * torch.from_numpy(np.array((0.229, 0.224, 0.225))).to(content.device) + torch.from_numpy(np.array((0.485, 0.456, 0.406))).to(content.device)
        style = style * torch.from_numpy(np.array((0.229, 0.224, 0.225))).to(style.device) + torch.from_numpy(np.array((0.485, 0.456, 0.406))).to(style.device)
        trs_img = trs_img * torch.from_numpy(np.array((0.229, 0.224, 0.225))).to(trs_img.device) + torch.from_numpy(np.array((0.485, 0.456, 0.406))).to(trs_img.device)
        
        content = content.permute(0, 3, 1, 2).float()
        style = style.permute(0, 3, 1, 2).float()
        trs_img = trs_img.permute(0, 3, 1, 2).float()
        
        # loss
        content_feats_vgg = self.encode_vgg_content(content)
        style_feats_vgg = self.encode_with_vgg_intermediate(style)
        trs_feats_vgg = self.encode_with_vgg_intermediate(trs_img)

        loss_c = self.calc_content_loss(trs_feats_vgg[-2], content_feats_vgg)
        loss_s = self.efdm_single(trs_feats_vgg[0], style_feats_vgg[0])
        for i in range(1, self.n_layers):
            loss_s = loss_s + self.efdm_single(trs_feats_vgg[i], style_feats_vgg[i])

        loss = loss_c * self.config.lambda_perc_cont + loss_s * self.config.lambda_perc_style

        # EFDM negative pair
        neg_idx = []
        batch = content.shape[0]
        for a in range(batch):
            neg_lst = {}
            for b in range(batch):  # for each image pair
                if a != b:
                    loss_s_single = 0
                    for i in range(0, self.n_layers):  # for each vgg layer
                        loss_s_single += self.efdm_single(trs_feats_vgg[i][a].unsqueeze(0), style_feats_vgg[i][b].unsqueeze(0))
                    neg_lst[b] = loss_s_single
            neg_lst = sorted(neg_lst, key=neg_lst.get)
            # neg_idx.append(neg_lst[:3])
            neg_idx.append([neg_lst[0]])

        return loss, neg_idx