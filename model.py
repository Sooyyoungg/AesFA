import torch
from torch import nn
import networks
import blocks
import time

from vgg19 import vgg, VGG_loss
from networks import EFDM_loss

class AesFA(nn.Module):
    def __init__(self, config):
        super(AesFA, self).__init__()

        self.config = config
        self.device = self.config.device

        self.lr = config.lr
        self.lambda_percept = config.lambda_percept
        self.lambda_const_style = config.lambda_const_style

        self.netE = networks.define_network(net_type='Encoder', config = config)    # Content Encoder
        self.netS = networks.define_network(net_type='Encoder', config = config)    # Style Encoder
        self.netG = networks.define_network(net_type='Generator', config = config)

        self.vgg_loss = VGG_loss(config, vgg)
        self.efdm_loss = EFDM_loss()

        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))

        self.E_scheduler = blocks.get_scheduler(self.optimizer_E, config)
        self.S_scheduler = blocks.get_scheduler(self.optimizer_S, config)
        self.G_scheduler = blocks.get_scheduler(self.optimizer_G, config)

        
    def forward(self, data):
        self.real_A = data['content_img'].to(self.device)
        self.real_B = data['style_img'].to(self.device)
       
        self.content_A, _, _ = self.netE(self.real_A)
        _, self.style_B, self.content_B_feat = self.netS(self.real_B)
        self.style_B_feat = self.content_B_feat.copy()
        self.style_B_feat.append(self.style_B)
        
        self.trs_AtoB, self.trs_AtoB_high, self.trs_AtoB_low = self.netG(self.content_A, self.style_B)

        self.trs_AtoB_content, _, self.content_trs_AtoB_feat = self.netE(self.trs_AtoB)
        _, self.trs_AtoB_style, self.style_trs_AtoB_feat = self.netS(self.trs_AtoB)
        self.style_trs_AtoB_feat.append(self.trs_AtoB_style)

        
    def calc_G_loss(self):
        self.G_percept, self.neg_idx = self.vgg_loss.perceptual_loss(self.real_A, self.real_B, self.trs_AtoB)
        self.G_percept *= self.lambda_percept
        
        self.G_contrast = self.efdm_loss(self.content_B_feat, self.style_B_feat, self.content_trs_AtoB_feat, self.style_trs_AtoB_feat, self.neg_idx) * self.lambda_const_style

        self.G_loss = self.G_percept + self.G_contrast

        
    def train_step(self, data):
        self.set_requires_grad([self.netE, self.netS, self.netG], True)

        self.forward(data)
        self.calc_G_loss()

        self.optimizer_E.zero_grad()
        self.optimizer_S.zero_grad()
        self.optimizer_G.zero_grad()
        self.G_loss.backward()
        self.optimizer_E.step()
        self.optimizer_S.step()
        self.optimizer_G.step()

        train_dict = {}
        train_dict['G_loss'] = self.G_loss
        train_dict['G_Percept'] = self.G_percept
        train_dict['G_Contrast'] = self.G_contrast
        
        train_dict['style_img'] = self.real_B
        train_dict['fake_AtoB'] = self.trs_AtoB
        train_dict['fake_AtoB_high'] = self.trs_AtoB_high
        train_dict['fake_AtoB_low'] = self.trs_AtoB_low
        
        return train_dict
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

                    
class AesFA_test(nn.Module):
    def __init__(self, config):
        super(AesFA_test, self).__init__()
        
        self.netE = networks.define_network(net_type='Encoder', config=config)
        self.netS = networks.define_network(net_type='Encoder', config=config)
        self.netG = networks.define_network(net_type='Generator', config=config)

    def forward(self, real_A, real_B, freq):
        with torch.no_grad():
            start = time.time()
            content_A = self.netE.forward_test(real_A, 'content')
            style_B = self.netS.forward_test(real_B, 'style')
            if freq:
                trs_AtoB, trs_AtoB_high, trs_AtoB_low = self.netG(content_A, style_B)
                end = time.time()
                during = end - start
                return trs_AtoB, trs_AtoB_high, trs_AtoB_low, during
            else:
                trs_AtoB = self.netG.forward_test(content_A, style_B)
                end = time.time()
                during = end - start
                return trs_AtoB, during
    
    def style_blending(self, real_A, real_B_1, real_B_2):
        with torch.no_grad():
            start = time.time()
            content_A = self.netE.forward_test(real_A, 'content')
            style_B1_h = self.netS.forward_test(real_B_1, 'style')[0]
            style_B2_l = self.netS.forward_test(real_B_2, 'style')[1]
            style_B = style_B1_h, style_B2_l

            trs_AtoB = self.netG.forward_test(content_A, style_B)
            end = time.time()
            during = end - start
            
        return trs_AtoB, during