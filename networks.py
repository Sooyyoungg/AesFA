import torch
from torch import nn
import torch.nn.functional as F

from blocks import *

def define_network(net_type, config = None):
    net = None
    alpha_in = config.alpha_in
    alpha_out = config.alpha_out
    sk = config.style_kernel

    if net_type == 'Encoder':
        net = Encoder(in_dim=config.input_nc, nf=config.nf, style_kernel=[sk, sk], alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Generator':
        net = Decoder(nf=config.nf, out_dim=config.output_nc, style_channel=256, style_kernel=[sk, sk, 3], alpha_in=alpha_in, freq_ratio=config.freq_ratio, alpha_out=alpha_out)
    return net

class Encoder(nn.Module):    
    def __init__(self, in_dim, nf=64, style_kernel=[3, 3], alpha_in=0.5, alpha_out=0.5):
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3)        
        
        self.OctConv1_1 = OctConv(in_channels=nf, out_channels=nf, kernel_size=3, stride=2, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="first")       
        self.OctConv1_2 = OctConv(in_channels=nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_3 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        self.OctConv2_1 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_3 = OctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        self.pool_h = nn.AdaptiveAvgPool2d((style_kernel[0], style_kernel[0]))
        self.pool_l = nn.AdaptiveAvgPool2d((style_kernel[1], style_kernel[1]))
        
        self.relu = Oct_conv_lreLU()

    def forward(self, x):   
        enc_feat = []
        out = self.conv(x)   
        
        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        out = self.OctConv1_3(out)
        out = self.relu(out)
        enc_feat.append(out)
        
        out = self.OctConv2_1(out)   
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        out = self.OctConv2_3(out)
        out = self.relu(out)
        enc_feat.append(out)
        
        out_high, out_low = out
        out_sty_h = self.pool_h(out_high)
        out_sty_l = self.pool_l(out_low)
        out_sty = out_sty_h, out_sty_l

        return out, out_sty, enc_feat
    
    def forward_test(self, x, cond):
        out = self.conv(x)   
        
        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        out = self.OctConv1_3(out)
        out = self.relu(out)
        
        out = self.OctConv2_1(out)   
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        out = self.OctConv2_3(out)
        out = self.relu(out)
        
        if cond == 'style':
            out_high, out_low = out
            out_sty_h = self.pool_h(out_high)
            out_sty_l = self.pool_l(out_low)
            return out_sty_h, out_sty_l
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, nf=64, out_dim=3, style_channel=512, style_kernel=[3, 3, 3], alpha_in=0.5, alpha_out=0.5, freq_ratio=[1,1], pad_type='reflect'):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        self.up_oct = Oct_conv_up(scale_factor=2)

        self.AdaOctConv1_1 = AdaOctConv(in_channels=4*nf, out_channels=4*nf, group_div=group_div[0], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=4*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_2 = OctConv(in_channels=4*nf, out_channels=2*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_1 = Oct_Conv_aftup(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out)

        self.AdaOctConv2_1 = AdaOctConv(in_channels=2*nf, out_channels=2*nf, group_div=group_div[1], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=2*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=2*nf, out_channels=nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_2 = Oct_Conv_aftup(nf, nf, 3, 1, 1, pad_type, alpha_in, alpha_out)

        self.AdaOctConv3_1 = AdaOctConv(in_channels=nf, out_channels=nf, group_div=group_div[2], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_2 = OctConv(in_channels=nf, out_channels=nf//2, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="last", freq_ratio=freq_ratio)
       
        self.conv4 = nn.Conv2d(in_channels=nf//2, out_channels=out_dim, kernel_size=1)

    def forward(self, content, style):        
        out = self.AdaOctConv1_1(content, style)
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style)
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        out = self.AdaOctConv3_1(out, style)
        out = self.OctConv3_2(out)
        out, out_high, out_low = out

        out = self.conv4(out)
        out_high = self.conv4(out_high)
        out_low = self.conv4(out_low)

        return out, out_high, out_low
    
    def forward_test(self, content, style):
        out = self.AdaOctConv1_1(content, style, 'test')
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style, 'test')
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)
     
        out = self.AdaOctConv3_1(out, style, 'test')
        out = self.OctConv3_2(out)

        out = self.conv4(out[0])
        return out


############## Contrastive Loss function ##############
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    mse_loss = nn.MSELoss()
    return mse_loss(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)

    loss = mse_loss(input_mean, target_mean) + \
            mse_loss(input_std, target_std)
    return loss

class EFDM_loss(nn.Module):
    def __init__(self):
        super(EFDM_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def efdm_single(self, style, trans):
        B, C, W, H = style.size(0), style.size(1), style.size(2), style.size(3)
        
        value_style, index_style = torch.sort(style.view(B, C, -1))
        value_trans, index_trans = torch.sort(trans.view(B, C, -1))
        inverse_index = index_trans.argsort(-1)
        
        return self.mse_loss(trans.view(B, C,-1), value_style.gather(-1, inverse_index))

    def forward(self, style_E, style_S, translate_E, translate_S, neg_idx):
        loss = 0.
        batch = style_E[0][0].shape[0]
        for b in range(batch):
            poss_loss = 0.
            neg_loss = 0.
        
            # Positive loss
            for i in range(len(style_E)):
                poss_loss += self.efdm_single(style_E[i][0][b].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_E[i][1][b].unsqueeze(0), translate_E[i][1][b].unsqueeze(0))
            for i in range(len(style_S)):
                poss_loss += self.efdm_single(style_S[i][0][b].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_S[i][1][b].unsqueeze(0), translate_S[i][1][b].unsqueeze(0))
                
            # Negative loss
            for nb in neg_idx[b]:
                for i in range(len(style_E)):
                    neg_loss += self.efdm_single(style_E[i][0][nb].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_E[i][1][nb].unsqueeze(0), translate_E[i][1][b].unsqueeze(0))
                for i in range(len(style_S)):
                    neg_loss += self.efdm_single(style_S[i][0][nb].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_S[i][1][nb].unsqueeze(0), translate_S[i][1][b].unsqueeze(0))

            loss += poss_loss / neg_loss

        return loss