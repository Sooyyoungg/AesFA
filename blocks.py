import os
import glob
from path import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from blocks import *

def model_save(ckpt_dir, model, optim_E, optim_S, optim_G, epoch, itr=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netE': model.netE.state_dict(),
                'netS': model.netS.state_dict(),
                'netG': model.netG.state_dict(),
                'optim_E': optim_E.state_dict(),
                'optim_S': optim_S.state_dict(),
                'optim_G': optim_G.state_dict()},
               '%s/model_iter_%d_epoch_%d.pth' % (ckpt_dir, itr+1, epoch+1))

def model_load(checkpoint, ckpt_dir, model, optim_E, optim_S, optim_G):
    if not os.path.exists(ckpt_dir):
        epoch = -1
        return model, optim_E, optim_S, optim_G, epoch
    
    ckpt_path = Path(ckpt_dir)
    if checkpoint:
        model_ckpt = ckpt_path + '/' + checkpoint
    else:
        ckpt_lst = ckpt_path.glob('model_iter_*')
        ckpt_lst.sort(key=lambda x: int(x.split('iter_')[1].split('_epoch')[0]))
        model_ckpt = ckpt_lst[-1]
    itr = int(model_ckpt.split('iter_')[1].split('_epoch_')[0])
    epoch = int(model_ckpt.split('iter_')[1].split('_epoch_')[1].split('.')[0])
    print(model_ckpt)

    dict_model = torch.load(model_ckpt)

    model.netE.load_state_dict(dict_model['netE'])
    model.netS.load_state_dict(dict_model['netS'])
    model.netG.load_state_dict(dict_model['netG'])
    optim_E.load_state_dict(dict_model['optim_E'])
    optim_S.load_state_dict(dict_model['optim_S'])
    optim_G.load_state_dict(dict_model['optim_G'])

    return model, optim_E, optim_S, optim_G, epoch, itr

def test_model_load(checkpoint, model):
    dict_model = torch.load(checkpoint)
    model.netE.load_state_dict(dict_model['netE'])
    model.netS.load_state_dict(dict_model['netS'])
    model.netG.load_state_dict(dict_model['netG'])
    return model

def get_scheduler(optimizer, config):
    if config.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.n_epoch - config.n_iter) / float(config.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

class Oct_Conv_aftup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type, alpha_in, alpha_out):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels*alpha_in)
        lf_out = int(out_channels*alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        self.conv_h = nn.Conv2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.conv_l = nn.Conv2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
    
    def forward(self, x):
        hf, lf = x
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        return hf, lf

class Oct_conv_reLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf
    
class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf

class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf


############## Encoder ##############
class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = [1, 1]):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        if type == 'first':
            self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, padding_mode=pad_type, bias = False)
            self.convl = nn.Conv2d(in_channels, lf_ch_out,
                                   kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        elif type == 'last':
            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        else:
            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False
            )
            if self.is_dw:
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d(
                    lf_ch_in, hf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
                self.H2L = nn.Conv2d(
                    hf_ch_in, lf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False
            )
            
    def forward(self, x):
        if self.type == 'first':
            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)
            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
            return output, out_h, out_l
        else:
            hf, lf = x
            if self.is_dw:
                hf, lf = self.H2H(hf), self.L2L(lf)
            else:
                hf, lf = self.H2H(hf) + self.L2H(self.upsample(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))
            return hf, lf
        

############## Decoder ##############
class AdaOctConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_div, style_channels, kernel_size,
                 stride, padding, oct_groups, alpha_in, alpha_out, type='normal'):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.type = type
        
        h_in = int(in_channels * (1 - self.alpha_in))
        l_in = in_channels - h_in

        n_groups_h = h_in // group_div
        n_groups_l = l_in // group_div
        
        style_channels_h = int(style_channels * (1 - self.alpha_in))
        style_channels_l = int(style_channels - style_channels_h)
        
        kernel_size_h = kernel_size[0]
        kernel_size_l = kernel_size[1]
        kernel_size_A = kernel_size[2]

        self.kernelPredictor_h = KernelPredictor(in_channels=h_in,
                                              out_channels=h_in,
                                              n_groups=n_groups_h,
                                              style_channels=style_channels_h,
                                              kernel_size=kernel_size_h)
        self.kernelPredictor_l = KernelPredictor(in_channels=l_in,
                                               out_channels=l_in,
                                               n_groups=n_groups_l,
                                               style_channels=style_channels_l,
                                               kernel_size=kernel_size_l)
        
        self.AdaConv_h = AdaConv2d(in_channels=h_in, out_channels=h_in, n_groups=n_groups_h)
        self.AdaConv_l = AdaConv2d(in_channels=l_in, out_channels=l_in, n_groups=n_groups_l)
        
        self.OctConv = OctConv(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size_A, stride=stride, padding=padding, groups=oct_groups,
                            alpha_in=alpha_in, alpha_out=alpha_out, type=type)
        
        self.relu = Oct_conv_lreLU()

    def forward(self, content, style, cond='train'):
        c_hf, c_lf = content
        s_hf, s_lf = style
        h_w_spatial, h_w_pointwise, h_bias = self.kernelPredictor_h(s_hf)
        l_w_spatial, l_w_pointwise, l_bias = self.kernelPredictor_l(s_lf)
        
        if cond == 'train':
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l

            output = self.relu(output)

            output = self.OctConv(output)
            if self.type != 'last':
                output = self.relu(output)
            return output
        
        if cond == 'test':
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l
            output = self.relu(output)
            output = self.OctConv(output)
            if self.type != 'last':
                output = self.relu(output)
            return output

class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super(KernelPredictor, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.w_channels = style_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(math.ceil(padding), math.ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)
        bias = self.bias(w)
        bias = bias.reshape(len(w), self.out_channels)
        return w_spatial, w_pointwise, bias

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(math.ceil(padding), math.floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        ys = []
        for i in range(len(x)):
            y = self.forward_single(x[i:i+1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x