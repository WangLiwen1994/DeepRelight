from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np

import torch.nn.utils.spectral_norm as spectral_norm

from torch.distributions import Normal


############################################################
### Functions
############################################################
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # nn.init.kaiming_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        # m.weight.data *= 0.1
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print('--------------------------------------------------------------')
    return num_params


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)

    def gaussian(x):
        return np.exp(
            (x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2

    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable conv we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    # conv img with a gaussian kernel that has been built with build_gauss_kernel
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


def define_BoundaryVAE(input_nc, output_nc, ngf, ndf, latent_variable_size, gpu_ids=[]):
    ##### BoundaryVAEv20
    netBVAE = BoundaryVAEv20(input_nc, output_nc, ngf, ndf, latent_variable_size)
    num_params = print_network(netBVAE)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netBVAE.cuda(gpu_ids[0])
    netBVAE.apply(weights_init)

    return netBVAE, num_params


def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance', gpu_ids=[]):
    netG = ImageTinker(input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=4, norm_layer=nn.BatchNorm2d,
                       pad_type='reflect')
    # netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsampling=5, n_blocks=9, norm_layer=nn.InstanceNorm2d, padding_type='reflect')

    num_params = print_network(netG)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)

    return netG, num_params


def define_B(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=3, norm='instance', gpu_ids=[]):
    netB = BlendGenerator(input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3, norm_layer=nn.InstanceNorm2d,
                          pad_type='reflect')

    num_params = print_network(netB)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netB.cuda(gpu_ids[0])
    netB.apply(weights_init)

    return netB, num_params


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    num_params = print_network(netD)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)

    return netD, num_params


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        # assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ELU()

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model += [nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 2), activation]
        model += [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 4), activation]
        model += [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 8), activation]
        model += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 8), activation]
        model += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 8), activation]

        model += [ResnetBlock_v2(ngf * 8, 3, 1, 1, 1, 1, True, 'reflect', 'instance', 'elu', False)]
        model += [ResnetBlock_v2(ngf * 8, 3, 1, 2, 2, 1, True, 'reflect', 'instance', 'elu', False)]
        model += [ResnetBlock_v2(ngf * 8, 3, 1, 3, 3, 1, True, 'reflect', 'instance', 'elu', False)]
        model += [NonLocalBlock(ngf * 8, sub_sample=False)]
        model += [ResnetBlock_v2(ngf * 8, 3, 1, 3, 3, 1, True, 'reflect', 'instance', 'elu', False)]
        model += [ResnetBlock_v2(ngf * 8, 3, 1, 2, 2, 1, True, 'reflect', 'instance', 'elu', False)]
        model += [ResnetBlock_v2(ngf * 8, 3, 1, 1, 1, 1, True, 'reflect', 'instance', 'elu', False)]

        model += [nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 8),
                  activation]
        model += [nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 8),
                  activation]
        model += [nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 4),
                  activation]
        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 2),
                  activation]
        model += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1), norm_layer(ngf), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        # model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # ### downsample
        # for i in range(n_downsampling):
        #     mult = 2**i
        #     model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #               norm_layer(ngf * mult * 2), activation]

        # ### resnet blocks
        # mult = 2**n_downsampling
        # for i in range(n_blocks):
        #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # ### upsample         
        # for i in range(n_downsampling):
        #     mult = 2**(n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
        #                norm_layer(int(ngf * mult / 2)), activation]
        # model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ImageTinker(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=4, norm_layer=nn.InstanceNorm2d,
                 pad_type='reflect', activation=nn.LeakyReLU(0.2, True)):
        assert (n_blocks >= 0)
        super(ImageTinker, self).__init__()

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d

        self.en_padd1 = self.pad(3)
        self.en_conv1 = nn.Conv2d(input_nc, ngf // 2, kernel_size=7, stride=1, padding=0)
        # self.en_norm1 = norm_layer(ngf // 2)
        self.en_acti1 = activation

        self.en_padd2 = self.pad(1)
        self.en_conv2 = nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=0)
        # self.en_norm2 = norm_layer(ngf)
        self.en_acti2 = activation

        self.en_padd3 = self.pad(1)
        self.en_conv3 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=0)
        # self.en_norm3 = norm_layer(ngf * 2)
        self.en_acti3 = activation

        self.en_padd4 = self.pad(1)
        self.en_conv4 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=0)
        # self.en_norm4 = norm_layer(ngf * 4)
        self.en_acti4 = activation

        self.md_mres1 = MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                                                 norm=None)
        self.md_mres2 = MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                                                 norm=None)
        self.md_mres3 = MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                                                 norm=None)
        self.md_mres4 = MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                                                 norm=None)
        self.md_mres5 = MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                                                 norm=None)
        self.md_mres6 = MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                                                 norm=None)

        self.md_satn1 = NonLocalBlock(ngf * 4, sub_sample=False, bn_layer=False)
        self.md_satn2 = NonLocalBlock(ngf * 2, sub_sample=False, bn_layer=False)

        self.de_upbi1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_padd1 = self.pad(1)
        self.de_conv1 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=0)
        # self.de_norm1 = norm_layer(ngf * 2)
        self.de_acti1 = activation

        self.de_mix_padd1 = self.pad(1)
        self.de_mix_conv1 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=0)
        # self.de_mix_norm1 = norm_layer(ngf * 2)
        self.de_mix_acti1 = activation

        self.de_lr_padd1 = self.pad(1)
        self.de_lr_conv1 = nn.Conv2d(ngf * 2, ngf // 2, kernel_size=3, stride=1, padding=0)
        # self.de_lr_norm1 = norm_layer(ngf // 2)
        self.de_lr_acti1 = activation

        self.de_lr_padd2 = self.pad(1)
        self.de_lr_conv2 = nn.Conv2d(ngf // 2, output_nc, kernel_size=3, stride=1, padding=0)
        # self.de_lr_acti2 = nn.Tanh()

        self.de_upbi2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_padd2 = self.pad(1)
        self.de_conv2 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=0)
        # self.de_norm2 = norm_layer(ngf)
        self.de_acti2 = activation

        self.de_upbi3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_padd3 = self.pad(1)
        self.de_conv3 = nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=0)
        # self.de_norm3 = norm_layer(ngf // 2)
        self.de_acti3 = activation

        self.de_padd4 = self.pad(3)
        self.de_conv4 = nn.Conv2d(ngf // 2, output_nc, kernel_size=7, stride=1, padding=0)
        # self.de_acti4 = nn.Tanh()

        self.de_padd4_1 = self.pad(1)
        self.de_conv4_1 = nn.Conv2d(ngf // 2, 1, kernel_size=3, stride=1, padding=0)
        self.de_acti4_1 = nn.Sigmoid()

        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.down = nn.UpsamplingBilinear2d(scale_factor=0.25)

    def forward(self, msked_img, msk, real_img=None):
        if real_img is not None:
            rimg = real_img
        else:
            rimg = msked_img
        x = torch.cat((msked_img, msk), dim=1)

        e1 = self.en_acti1(self.en_conv1(self.en_padd1(x)))
        e2 = self.en_acti2(self.en_conv2(self.en_padd2(e1)))
        e3 = self.en_acti3(self.en_conv3(self.en_padd3(e2)))
        e4 = self.en_acti4(self.en_conv4(self.en_padd4(e3)))

        # middle 
        m1 = self.md_mres1(e4)
        m2 = self.md_mres2(m1)
        m3 = self.md_mres3(m2)
        a1 = self.md_satn1(m3)
        m4 = self.md_mres4(a1)
        m5 = self.md_mres5(m4)
        m6 = self.md_mres6(m5)

        a2 = self.md_satn2(e3)

        # decode 
        d1 = self.de_acti1(self.de_conv1(self.de_padd1(self.de_upbi1(m6))))
        skp = torch.cat((d1, a2), dim=1)
        d2 = self.de_mix_acti1(self.de_mix_conv1(self.de_mix_padd1(skp)))

        lr1 = self.de_lr_acti1(self.de_lr_conv1(self.de_lr_padd1(d2)))
        lr2 = self.de_lr_conv2(self.de_lr_padd2(lr1))

        d3 = self.de_acti2(self.de_conv2(self.de_padd2(self.de_upbi2(d2))))
        d4 = self.de_acti3(self.de_conv3(self.de_padd3(self.de_upbi3(d3))))

        d5 = self.de_conv4(self.de_padd4(d4))
        d5_1 = self.de_acti4_1(self.de_conv4_1(self.de_padd4_1(d4)))

        lr_x = lr2
        lr_x2 = lr_x * self.down(msk) + self.down(rimg) * (1.0 - self.down(msk))

        compltd_img = d5
        compltd_img = compltd_img * msk + rimg * (1.0 - msk)
        lr_compltd_img = self.down(compltd_img)

        lr_res = lr_x2 - lr_compltd_img
        hr_res = self.up(lr_res)

        out = compltd_img + hr_res * d5_1

        return compltd_img, out, lr_x
        # return compltd_img, reconst_img, lr_x


class BlendGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3, norm_layer=nn.InstanceNorm2d,
                 pad_type='reflect', activation=nn.ELU()):
        assert (n_blocks >= 0)
        super(BlendGenerator, self).__init__()

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d

            # Image encode
        self.en_padd1 = self.pad(3)
        self.en_conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0)
        self.en_norm1 = norm_layer(ngf)
        self.en_acti1 = activation

        self.en_padd2 = self.pad(1)
        self.en_conv2 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=0)
        self.en_norm2 = norm_layer(ngf * 2)
        self.en_acti2 = activation

        self.en_padd3 = self.pad(1)
        self.en_conv3 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=0)
        self.en_norm3 = norm_layer(ngf * 4)
        self.en_acti3 = activation

        self.en_padd4 = self.pad(1)
        self.en_conv4 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=0)
        self.en_norm4 = norm_layer(ngf * 8)
        self.en_acti4 = activation

        # middle resnetblocks 
        self.res_blk1 = ResnetBlock(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='reflect', norm='instance')
        self.res_blk2 = ResnetBlock(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='reflect', norm='instance')
        self.res_blk3 = ResnetBlock(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='reflect', norm='instance')

        # image decoder 
        self.de_conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.de_norm1 = norm_layer(ngf * 4)
        self.de_acti1 = activation

        self.de_conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.de_norm2 = norm_layer(ngf * 2)
        self.de_acti2 = activation

        self.de_conv3 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.de_norm3 = norm_layer(ngf)
        self.de_acti3 = activation

        self.de_padd4 = self.pad(3)
        self.de_conv4 = nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0)
        self.de_acti4 = nn.Sigmoid()

    def forward(self, completed_img, msked_img):
        x = torch.cat((completed_img, msked_img), dim=1)
        e1 = self.en_acti1(self.en_norm1(self.en_conv1(self.en_padd1(x))))  # 512x512x64
        e2 = self.en_acti2(self.en_norm2(self.en_conv2(self.en_padd2(e1))))  # 256x256x128
        e3 = self.en_acti3(self.en_norm3(self.en_conv3(self.en_padd3(e2))))  # 128x128x256
        e4 = self.en_acti4(self.en_norm4(self.en_conv4(self.en_padd4(e3))))  # 64x64x512

        middle1 = self.res_blk1(e4)
        middle2 = self.res_blk2(middle1)
        middle3 = self.res_blk3(middle2)

        d1 = self.de_acti1(self.de_norm1(self.de_conv1(middle3)))  # 128x128x256
        d2 = self.de_acti2(self.de_norm2(self.de_conv2(d1)))  # 256x256x128
        d3 = self.de_acti3(self.de_norm3(self.de_conv3(d2)))  # 512x512x64
        d4 = self.de_acti4(self.de_conv4(self.de_padd4(d3)))  # 512x512x1

        return completed_img * d4 + msked_img * (1.0 - d4), d4

    ############################################################


### Losses
############################################################
class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.__tensor__size(x[:, :, 1:, :])
        count_w = self.__tensor__size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()
        self.epsilon = 1e-10

    def forward(self, pred, gt):
        # sigmoid_pred = torch.sigmoid(pred)

        count_pos = torch.sum(gt) * 1.0 + self.epsilon
        count_neg = torch.sum(1. - gt) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(pred, gt)

        return loss

    # Lap_criterion = LapLoss(max_levels=5)


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        self.L1_loss = nn.L1Loss()

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma,
                                                    n_channels=input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(self.L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


class LapMap(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapMap, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma,
                                                    n_channels=input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)

        return pyr_input


class VGGLoss(nn.Module):
    # vgg19 perceptual loss
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                    self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class DHingeLoss(nn.Module):
    # hinge loss for discriminator 
    def forward(self, x, target_is_real):
        # d_loss = 0
        # for input_i in x:
        #     pred = input_i[-1] 
        #     one_tensor = torch.FloatTensor(pred.size()).fill_(1) 
        #     one_tensor = Variable(one_tensor, requires_grad=False) 

        #     if target_is_real: 
        #         # d_loss_real
        #         d_loss += torch.nn.ReLU()(one_tensor - pred).mean() 
        #     else: 
        #         # d_loss_fake
        #         d_loss += torch.nn.ReLU()(one_tensor - pred).mean() 
        # return d_loss 
        zero_tensor = torch.FloatTensor(1).fill_(0)
        zero_tensor.requires_grad_(False)
        zero_tensor = zero_tensor.expand_as(x)
        if target_is_real:
            minval = torch.min(x - 1, zero_tensor)
            loss = -torch.mean(minval)
        else:
            minval = torch.min(-x - 1, zero_tensor)
            loss = -torch.mean(minval)


class GHingeLoss(nn.Module):
    # hinge loss for generator 
    # g_loss_fake
    def forward(self, x):
        # g_loss = 0 
        # for input_i in x:
        #     pred = input_i[-1] 
        #     one_tensor = torch.FloatTensor(pred.size()).fill_(1) 
        #     one_tensor = Variable(one_tensor, requires_grad=False) 

        #     g_loss += -torch.mean(x) 
        # return g_loss 
        return -x.mean()


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

        # Define the PatchGAN discriminator with the specified arguments.


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                # nn.LeakyReLU(0.2, True)
                # norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            # norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[SpectralNorm(nn.Conv2d(nf, nf, kernel_size=kw, stride=1, padding=padw))]]
        # sequence += [[SpectralNorm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))]]
        # sequence += [[MultiDilationResnetBlock_v2(nf, kernel_size=3, stride=1, padding=1)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

        # Define the Multiscale Discriminator.


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3,
                 getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

    ### Define Vgg19 for vgg_loss


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(1):
            # relu1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):
            # relu2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):
            # relu3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):
            # relu4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):
            # relu5_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # fixed pretrained vgg19 model for feature extraction
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    ### Multi-Dilation ResnetBlock


class MultiDilationResnetBlock(nn.Module):
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 pad_type='reflect', norm='instance', acti='relu', use_dropout=False):
        super(MultiDilationResnetBlock, self).__init__()
        # self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout)

        ### hard code, 4 dilation levels 
        self.branch1 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=2, dilation=2, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch2 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=3, dilation=3, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch3 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=4, dilation=4, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch4 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=5, dilation=5, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch5 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=6, dilation=6, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch6 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=8, dilation=8, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch7 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=10, dilation=10, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch8 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=12, dilation=12, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')

        self.fusion9 = ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti=None)

        # def build_conv_block(self, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout):

    #     conv_block = []
    #     conv_block += [ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti='relu')]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)] 
    #     conv_block += [ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti=None)] 

    #     return nn.Sequential(*conv_block) 

    def forward(self, x):
        d1 = self.branch1(x)
        d2 = self.branch2(x)
        d3 = self.branch3(x)
        d4 = self.branch4(x)
        d5 = self.branch5(x)
        d6 = self.branch6(x)
        d7 = self.branch7(x)
        d8 = self.branch8(x)
        d9 = torch.cat((d1, d2, d3, d4, d5, d6, d7, d8), dim=1)
        out = x + self.fusion9(d9)
        return out

    ### Multi-Dilation ResnetBlock


class MultiDilationResnetBlock_v2(nn.Module):
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 pad_type='reflect', norm='instance', acti='relu', use_dropout=False):
        super(MultiDilationResnetBlock_v2, self).__init__()
        # self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout)

        ### hard code, 4 dilation levels 
        self.branch1 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=2, dilation=2, groups=1,
                                 bias=True, pad_type=pad_type, norm='spectral', acti='relu')
        self.branch2 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=4, dilation=4, groups=1,
                                 bias=True, pad_type=pad_type, norm='spectral', acti='relu')
        self.branch3 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=8, dilation=8, groups=1,
                                 bias=True, pad_type=pad_type, norm='spectral', acti='relu')
        self.branch4 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=12, dilation=12, groups=1,
                                 bias=True, pad_type=pad_type, norm='spectral', acti='relu')

        self.fusion5 = ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                                 bias=True, pad_type=pad_type, norm='spectral', acti=None)
        self.shrtcut = ConvBlock(input_nc, input_nc, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                 bias=True, pad_type=pad_type, norm='spectral', acti=None)

    # def build_conv_block(self, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout):
    #     conv_block = [] 
    #     conv_block += [ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti='relu')]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)] 
    #     conv_block += [ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti=None)] 

    #     return nn.Sequential(*conv_block) 

    def forward(self, x):
        d1 = self.branch1(x)
        d2 = self.branch2(x)
        d3 = self.branch3(x)
        d4 = self.branch4(x)
        d5 = torch.cat((d1, d2, d3, d4), dim=1)
        out = self.shrtcut(x) + self.fusion5(d5)
        return out


from .base_model_liwen import FusionLayer


class MultiDilationResnetBlock_attention(nn.Module):
    def __init__(self, input_nc_each, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 pad_type='reflect', norm='instance', acti='relu', use_dropout=False):
        super(MultiDilationResnetBlock_attention, self).__init__()
        # self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout)

        ### hard code, 4 dilation levels
        input_nc = input_nc_each * 2
        self.branch1 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=2, dilation=2, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch2 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=3, dilation=3, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch3 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=4, dilation=4, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch4 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=5, dilation=5, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch5 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=6, dilation=6, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch6 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=8, dilation=8, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch7 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=10, dilation=10, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')
        self.branch8 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=12, dilation=12, groups=1,
                                 bias=True, pad_type=pad_type, norm=norm, acti='relu')

        # self.fusion9 = ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
        #                          bias=True, pad_type=pad_type, norm=norm, acti=None)
        self.fusion = FusionLayer(inchannel=input_nc, outchannel=input_nc_each, reduction=8)

    def forward(self, x_hdr, x_relight):
        x = torch.cat([x_hdr, x_relight], dim=1)

        d1 = self.branch1(x)
        d2 = self.branch2(x)
        d3 = self.branch3(x)
        d4 = self.branch4(x)
        d5 = self.branch5(x)
        d6 = self.branch6(x)
        d7 = self.branch7(x)
        d8 = self.branch8(x)

        d9 = torch.cat((d1, d2, d3, d4, d5, d6, d7, d8), dim=1)

        out = x_relight + self.fusion(d9)
        return out

    ### ResnetBlock


class ResnetBlock(nn.Module):
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 pad_type='reflect', norm='instance', acti='relu', use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias,
                                                pad_type, norm, acti, use_dropout)

    def build_conv_block(self, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti,
                         use_dropout):
        conv_block = []
        conv_block += [
            ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm,
                      acti='relu')]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm,
                      acti=None)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    ### ResnetBlock


class ResnetBlock_v2(nn.Module):
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 pad_type='reflect', norm='instance', acti='relu', use_dropout=False):
        super(ResnetBlock_v2, self).__init__()
        self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias,
                                                pad_type, norm, acti, use_dropout)

    def build_conv_block(self, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti,
                         use_dropout):
        conv_block = []
        conv_block += [
            ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=padding, dilation=dilation, groups=groups,
                      bias=bias, pad_type=pad_type, norm=norm, acti='elu')]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            ConvBlock(input_nc, input_nc, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True,
                      pad_type='reflect', norm='instance', acti=None)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    ### SPADEResnetBlock


class SPADEResnetBlock(nn.Module):
    def __init__(self, s_input_nc, input_nc, output_nc, scale_factor, norm='spectral'):
        super(SPADEResnetBlock, self).__init__()
        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        # create conv layers 
        self.conv_0 = nn.Conv2d(input_nc, middle_nc, 3, 1, 1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, 3, 1, 1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, 1, 1, 0, bias=False)

        if 'spectral' in norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

                # define normalization layers
        self.norm_0 = SPADE(s_input_nc, input_nc, 3, scale_factor=scale_factor, norm='instance')
        self.norm_1 = SPADE(s_input_nc, middle_nc, 3, scale_factor=scale_factor, norm='instance')
        if self.learned_shortcut:
            self.norm_s = SPADE(s_input_nc, input_nc, 3, scale_factor=scale_factor, norm='instance')

        self.acti = nn.LeakyReLU(0.2, False)

    def forward(self, x_featmap, c_featmap):
        x_featmap_s = self.shortcut(x_featmap, c_featmap)

        dx = self.conv_0(self.norm_0(x_featmap, c_featmap))
        dx = self.conv_1(self.norm_1(dx, c_featmap))

        out = x_featmap_s + dx
        return out

    def shortcut(self, x_featmap, c_featmap):
        if self.learned_shortcut:
            x_featmap_s = self.conv_s(self.norm_s(x_featmap, c_featmap))
        else:
            x_featmap_s = x_featmap
        return x_featmap_s

    ### GatedSPADEResnetBlock


class GatedSPADEResnetBlock(nn.Module):
    def __init__(self, s_input_nc, input_nc, output_nc, scale_factor, norm='spectral'):
        super(GatedSPADEResnetBlock, self).__init__()
        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        # create conv layers 
        self.conv_0 = nn.Conv2d(input_nc, middle_nc, 3, 1, 1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, 3, 1, 1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, 1, 1, 0, bias=False)

        if 'spectral' in norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

                # define normalization layers
        self.norm_0 = GatedSPADE(s_input_nc, input_nc, 3, scale_factor=scale_factor, norm='instance')
        self.norm_1 = GatedSPADE(s_input_nc, middle_nc, 3, scale_factor=scale_factor, norm='instance')
        # self.norm_0 = SPADE(s_input_nc, input_nc, 3, scale_factor=scale_factor, norm='instance')
        # self.norm_1 = SPADE(s_input_nc, middle_nc, 3, scale_factor=scale_factor, norm='instance')
        if self.learned_shortcut:
            self.norm_s = GatedSPADE(s_input_nc, input_nc, 3, scale_factor=scale_factor, norm='instance')
            # self.norm_s = SPADE(s_input_nc, input_nc, 3, scale_factor=scale_factor, norm='instance')

        self.acti = nn.LeakyReLU(0.2, False)

    def forward(self, x_featmap, c_featmap):
        x_featmap_s = self.shortcut(x_featmap, c_featmap)

        dx = self.conv_0(self.acti(self.norm_0(x_featmap, c_featmap)))
        dx = self.conv_1(self.acti(self.norm_1(dx, c_featmap)))

        out = x_featmap_s + dx
        return out

    def shortcut(self, x_featmap, c_featmap):
        if self.learned_shortcut:
            x_featmap_s = self.conv_s(self.acti(self.norm_s(x_featmap, c_featmap)))
        else:
            x_featmap_s = x_featmap
        return x_featmap_s

    ### BackProjectionBlock


class BackPrjBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm='instance'):
        super(BackPrjBlock, self).__init__()

        # create conv layers 
        self.conv_0 = ConvBlock(input_nc, output_nc, 3, 1, 1, norm=norm, acti='lrelu')
        self.conv_1 = ConvBlock(output_nc, input_nc, 3, 1, 1, norm=norm, acti='lrelu')
        self.conv_2 = ConvBlock(input_nc, output_nc, 3, 1, 1, norm=norm, acti='lrelu')

    def forward(self, x):
        d1 = self.conv_0(x)
        u1 = self.conv_1(d1)

        d2 = self.conv_2(x - u1)

        return d1 + d2

    ### PyramidAttentionBlock


class PyrAttnBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=2, pyr=2, gated=True, pad_type='reflect',
                 norm='instance', acti='lrelu'):
        super(PyrAttnBlock, self).__init__()
        self.use_gatedconv = gated
        self.pyr = pyr  ### pyr should be an even number. i.e. 2, 4, 6
        conv_block = []

        for i in range(pyr):
            padw = i + 1
            dilr = i + 1
            if gated:
                conv_block += [[GatedConvBlock(input_nc, output_nc // pyr, kernel_size, stride, padding=padw,
                                               dilation=dilr, pad_type=pad_type, norm=norm, acti=acti)]]
            else:
                conv_block += [[ConvBlock(input_nc, output_nc // pyr, kernel_size, stride, padding=padw, dilation=dilr,
                                          pad_type=pad_type, norm=norm, acti=acti)]]

        for n in range(len(conv_block)):
            setattr(self, 'branch' + str(n), nn.Sequential(*conv_block[n]))

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling layer
        self.sq_conv = ConvBlock(output_nc, output_nc // 2, 1, 1, acti='relu')
        self.ex_conv = ConvBlock(output_nc // 2, output_nc, 1, 1, acti='sigmoid')

    def forward(self, input):
        # concat 
        for n in range(self.pyr):
            model = getattr(self, 'branch' + str(n))
            # res.append(model(input))
            out = model(input)
            if n == 0:
                res = out.clone()
            else:
                res = torch.cat((res, out), dim=1)

                # channel weighting
        w_v = self.ex_conv(self.sq_conv(self.gap(res)))
        out = torch.mul(w_v.expand_as(res), res)

        return out

    ### NonLocalBlock2D


class NonLocalBlock(nn.Module):
    def __init__(self, input_nc, inter_nc=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()
        self.input_nc = input_nc
        self.inter_nc = inter_nc

        if inter_nc is None:
            self.inter_nc = input_nc // 2

        self.g = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_nc, out_channels=self.input_nc, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.input_nc)
            )
            self.W[0].weight.data.zero_()
            self.W[0].bias.data.zero_()
        else:
            self.W = nn.Conv2d(in_channels=self.inter_nc, out_channels=self.input_nc, kernel_size=1, stride=1,
                               padding=0)
            self.W.weight.data.zero_()
            self.W.bias.data.zero_()

        self.theta = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size(2, 2)))

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)

        z = W_y + x
        return z

    ### NonLocalBlock2D


class SABlock(nn.Module):
    def __init__(self, input_nc, inter_nc=None, sub_sample=True, bn_layer=True):
        super(SABlock, self).__init__()
        self.input_nc = input_nc
        self.inter_nc = inter_nc

        if inter_nc is None:
            self.inter_nc = input_nc // 2

        self.g = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_nc, out_channels=self.input_nc, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(self.input_nc)
            )
            self.W[0].weight.data.zero_()
            self.W[0].bias.data.zero_()
        else:
            self.W = nn.Conv2d(in_channels=self.inter_nc, out_channels=self.input_nc, kernel_size=1, stride=1,
                               padding=0)
            self.W.weight.data.zero_()
            self.W.bias.data.zero_()

        self.theta = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size(2, 2)))

    def forward(self, x, x2):
        batch_size = x.size(0)

        g_x = self.g(x2).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)

        z = W_y + x
        return z

    ### ConvBlock


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 pad_type='zero', norm=None, acti='lrelu'):
        super(ConvBlock, self).__init__()
        self.use_bias = bias

        # initialize padding 
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

            # initialize normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(output_nc)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_nc)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_nc)
        elif norm is None or norm == 'spectral':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

            # initialize activation
        if acti == 'relu':
            self.acti = nn.ReLU(inplace=True)
        elif acti == 'lrelu':
            self.acti = nn.LeakyReLU(0.2, inplace=True)
        elif acti == 'prelu':
            self.acti = nn.PReLU()
        elif acti == 'elu':
            self.acti = nn.ELU()
        elif acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'sigmoid':
            self.acti = nn.Sigmoid()
        elif acti is None:
            self.acti = None
        else:
            assert 0, "Unsupported activation: {}".format(acti)

            # initialize convolution
        if norm == 'spectral':
            self.conv = SpectralNorm(
                nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups,
                          bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups,
                                  bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.acti:
            x = self.acti(x)
        return x

    ### GatedConvBlock


class GatedConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 pad_type='zero', norm=None, acti='lrelu'):
        super(GatedConvBlock, self).__init__()
        self.use_bias = bias

        # initialize padding 
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

            # initialize normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(output_nc)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_nc)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_nc)
        elif norm is None or norm == 'spectral':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

            # initialize activation
        if acti == 'relu':
            self.acti = nn.ReLU(inplace=True)
        elif acti == 'lrelu':
            self.acti = nn.LeakyReLU(0.2, inplace=True)
        elif acti == 'prelu':
            self.acti = nn.PReLU()
        elif acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'sigmoid':
            self.acti = nn.Sigmoid()
        elif acti is None:
            self.acti = None
        else:
            assert 0, "Unsupported activation: {}".format(acti)

        self.gate_acti = nn.Sigmoid()

        # initialize convolution 
        if norm == 'spectral':
            self.conv = SpectralNorm(
                nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups,
                          bias=self.use_bias))
            self.gate = SpectralNorm(
                nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups,
                          bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups,
                                  bias=self.use_bias)
            self.gate = nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups,
                                  bias=self.use_bias)

    def forward(self, x):
        inp = x.clone()
        f = self.conv(self.pad(x))
        g = self.gate_acti(self.gate(self.pad(inp)))
        # gf = torch.mul(f, g)
        gf = f * g
        if self.norm:
            gf = self.norm(gf)
        if self.acti:
            gf = self.acti(gf)
        return gf

    ### LinearBlock


class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm=None, acti='lrelu'):
        super(LinearBlock, self).__init__()
        self.use_bias = True

        # initialize fully connected layer 
        if norm == 'spectral':
            self.fc = SpectralNorm(nn.Linear(input_nc, output_nc, bias=self.use_bias))
        else:
            self.fc = nn.Linear(input_nc, output_nc, bias=self.use_bias)

            # initialize normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(output_nc)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(output_nc)
        elif norm is None or norm == 'spectral':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

            # initialize activation
        if acti == 'relu':
            self.acti = nn.ReLU(inplace=True)
        elif acti == 'lrelu':
            self.acti = nn.LeakyReLU(0.2, inplace=True)
        elif acti == 'prelu':
            self.acti = nn.PReLU()
        elif acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'sigmoid':
            self.acti = nn.Sigmoid()
        elif acti is None:
            self.acti = None
        else:
            assert 0, "Unsupported activation: {}".format(acti)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.acti:
            out = self.acti(out)
        return out

    ### AdaIN


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned 
        self.weight = None
        self.bias = None
        # just dummy buffers, not used 
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # apply instance norm 
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

        # ######### put the following two functions into the model #########
    # def assign_adain_params(self, adain_params, model):
    #     # assign the adain_params to the AdaIN layers in model 
    #     for m in model.modules():
    #         if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
    #             mean = adain_params[:, :m.num_features]
    #             std = adain_params[:, m.num_features:2*m.num_features] 
    #             m.bias = mean.contiguous().view(-1)
    #             m.weight = std.contiguous.view(-1) 
    #             if adain_params.size(1) > 2*m.num_features:
    #                 adain_params = adain_params[:, 2*m.num_features:] 

    # def get_num_adain_params(self, model):
    #     # return the number of AdaIN parameters needed by the model 
    #     num_adain_params = 0 
    #     for m in model.modules(): 
    #         if m.__class__.__name__ == "AdaptiveInstanceNorm2d": 
    #             num_adain_params += 2*m.num_features 
    #     return num_adain_params 
    # ######### put the above two functions into the model #########


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


### SpectralNorm 
class SpectralNorm(nn.Module):
    """
    Spectral Normalization for Generative Adversarial Networks
    Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan 
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    # Define the BoundaryVAEv2


class BoundaryVAEv2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = ConvBlock(self.RGB, ndf, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu')

        ### masked image encoder (still use during testing) 
        self.mi_e1 = GatedConvBlock(input_nc, ndf, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e2 = GatedConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e3 = GatedConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e4 = GatedConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu')

        ### shared encoder and vae encoder (not use during testing) 
        self.shrd_e_SPADE1 = SPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')
        self.shrd_e1 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')
        self.shrd_e2 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')

        self.vae_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.vae_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar
        self.vae_fc3 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # x_i
        self.vae_fc4 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # y_i

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d2 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d3 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')
        self.vae_d_SPADE1 = SPADE(ngf * 8, ngf * 8, 3, 1, 1, norm='instance')

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d4 = ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d5 = ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d6 = ConvBlock(ngf * 2, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d7 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='sigmoid')

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        mu = logvar = x_i = y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32,
                                              device=x.get_device())

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)

            h5 = self.shrd_e_SPADE1(rl_h4, msk_h4)
            h6 = self.shrd_e1(h5)
            h7 = self.shrd_e2(h6)

            h7 = h7.view(-1, self.ndf * 8 * 8 * 8)
            mu = self.vae_fc1(h7)
            logvar = self.vae_fc2(h7)
            x_i = self.vae_fc3(h7)
            y_i = self.vae_fc4(h7)

        return msk_h4, mu, logvar, x_i, y_i

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, msk_img_feat, z):
        h1 = self.vae_d1(z)
        h1 = h1.view(-1, self.ngf * 8, 8, 8)

        h2 = self.vae_d2(self.up1(h1))
        h3 = self.vae_d3(self.up2(h2))
        h3 = self.vae_d_SPADE1(h3, msk_img_feat)
        h4 = self.vae_d4(self.up3(h3))
        h5 = self.vae_d5(self.up4(h4))
        h6 = self.vae_d6(self.up5(h5))

        return self.vae_d7(self.up6(h6))

    def forward(self, x, target=None):
        msk_img_feat, mu, logvar, x_i, y_i = self.encode(x, target)
        if target is not None:
            z = self.reparametrize(mu, logvar)
            new_y_i = self.reparametrize(y_i, x_i)
            new_x_i = self.reparametrize(y_i, x_i)
        else:
            z = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            new_y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            new_x_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        adain_params = torch.cat((new_y_i, new_x_i), dim=1)
        self.assign_adain_params(adain_params, self)
        res = self.decode(msk_img_feat, z)

        return res, mu, logvar, x_i, y_i

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv3


class BoundaryVAEv3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = ConvBlock(self.RGB, ndf, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu')

        ### masked image encoder (still use during testing) 
        self.mi_e1 = GatedConvBlock(input_nc, ndf, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e2 = GatedConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e3 = GatedConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e4 = GatedConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu')

        ### shared encoder and vae encoder (not use during testing) 
        # self.shrd_e_SPADE1 = SPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')
        self.shrd_e1 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')
        self.shrd_e2 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')

        self.vae_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.vae_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar
        # self.vae_fc3 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # x_i
        # self.vae_fc4 = LinearBlock(ndf *8*8*8, latent_variable_size, norm='batch', acti=None) # y_i

        ### vae decoder (still use during testing) 
        # self.vae_d1 = LinearBlock(latent_variable_size, ngf *8*8*8, norm=None, acti=None)
        self.vae_d1 = LinearBlock(latent_variable_size, latent_variable_size * 2, norm=None, acti=None)

        self.adain_layer = AdaptiveInstanceNorm2d(latent_variable_size)

        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d2 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')

        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d3 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')
        # self.vae_d_SPADE1 = SPADE(ngf * 8, ngf * 8, 3, 1, 1, norm='instance')

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d4 = ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d5 = ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d6 = ConvBlock(ngf * 2, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d7 = ConvBlock(ngf, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.vae_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='sigmoid')

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        mu = logvar = x_i = y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32,
                                              device=x.get_device())

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)

            # h5 = self.shrd_e_SPADE1(rl_h4, msk_h4)
            h5 = rl_h4 + msk_h4
            h6 = self.shrd_e1(h5)
            h7 = self.shrd_e2(h6)

            h7 = h7.view(-1, self.ndf * 8 * 8 * 8)
            mu = self.vae_fc1(h7)
            logvar = self.vae_fc2(h7)
            # x_i = self.vae_fc3(h7)
            # y_i = self.vae_fc4(h7)

        return msk_h1, msk_h2, msk_h3, msk_h4, mu, logvar, x_i, y_i

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, msk_h1, msk_h2, msk_h3, msk_img_feat, z):
        h1 = self.vae_d1(z)
        # h1 = h1.view(-1, self.ngf *8, 8, 8) 
        self.assign_adain_params(h1, self)

        h3 = self.adain_layer(msk_img_feat)
        # h2 = self.vae_d2(self.up1(h1)) 
        # h3 = self.vae_d3(self.up2(h2)) 
        # h3 = self.vae_d_SPADE1(h3, msk_img_feat) 
        h4 = self.vae_d4(self.up3(h3))
        h5 = self.vae_d5(self.up4(h4 + msk_h3))
        h6 = self.vae_d6(self.up5(h5 + msk_h2))
        h7 = self.vae_d7(self.up6(h6 + msk_h1))

        return self.vae_d8(h7)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_img_feat, mu, logvar, x_i, y_i = self.encode(x, target)
        if target is not None:
            z = self.reparametrize(mu, logvar)
            # new_y_i = self.reparametrize(y_i, x_i)
            # new_x_i = self.reparametrize(y_i, x_i)
        else:
            z = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_x_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        # adain_params = torch.cat((new_y_i, new_x_i), dim=1)
        # self.assign_adain_params(adain_params, self)
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_img_feat, z)

        return res, mu, logvar, x_i, y_i

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv4


class BoundaryVAEv4(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv4, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = PyrAttnBlock(self.RGB, ndf, 3, 2, 2, False)
        self.ri_e2 = PyrAttnBlock(ndf, ndf * 2, 3, 2, 2, False)
        self.ri_e3 = PyrAttnBlock(ndf * 2, ndf * 4, 3, 2, 2, False)
        self.ri_e4 = PyrAttnBlock(ndf * 4, ndf * 8, 3, 2, 2, False)
        # self.ri_e1 = ConvBlock(self.RGB, ndf, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.ri_e2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu') 

        ### masked image encoder (still use during testing) 
        self.mi_e1 = PyrAttnBlock(input_nc, ndf, 3, 2, 2, True)
        self.mi_e2 = PyrAttnBlock(ndf, ndf * 2, 3, 2, 2, True)
        self.mi_e3 = PyrAttnBlock(ndf * 2, ndf * 4, 3, 2, 2, True)
        self.mi_e4 = PyrAttnBlock(ndf * 4, ndf * 8, 3, 2, 2, True)
        # self.mi_e1 = GatedConvBlock(input_nc, ndf, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.mi_e2 = GatedConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.mi_e3 = GatedConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.mi_e4 = GatedConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu') 

        ### shared encoder and vae encoder (not use during testing) 
        # self.shrd_e_SPADE1 = SPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')
        self.shrd_e1 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')
        self.shrd_e2 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')

        self.vae_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.vae_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar
        # self.vae_fc3 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # x_i
        # self.vae_fc4 = LinearBlock(ndf *8*8*8, latent_variable_size, norm='batch', acti=None) # y_i

        ### vae decoder (still use during testing) 
        # self.vae_d1 = LinearBlock(latent_variable_size, ngf *8*8*8, norm=None, acti=None)
        self.vae_d1 = LinearBlock(latent_variable_size, latent_variable_size * 2, norm=None, acti=None)
        self.vae_d1_2 = LinearBlock(latent_variable_size * 2, 128 * 2, norm=None, acti=None)
        self.vae_d1_3 = LinearBlock(128 * 2, 64 * 2, norm=None, acti=None)
        self.vae_d1_4 = LinearBlock(64 * 2, 32 * 2, norm=None, acti=None)

        self.adain_layer = AdaptiveInstanceNorm2d(latent_variable_size)

        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d2 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')

        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d3 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')
        # self.vae_d_SPADE1 = SPADE(ngf * 8, ngf * 8, 3, 1, 1, norm='instance')

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d4 = ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d5 = ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d6 = ConvBlock(ngf * 2, ngf, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d7 = ConvBlock(ngf, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.vae_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='sigmoid')

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        mu = logvar = x_i = y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32,
                                              device=x.get_device())

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)

            # h5 = self.shrd_e_SPADE1(rl_h4, msk_h4)
            h5 = rl_h4 + msk_h4
            h6 = self.shrd_e1(h5)
            h7 = self.shrd_e2(h6)

            h7 = h7.view(-1, self.ndf * 8 * 8 * 8)
            mu = self.vae_fc1(h7)
            logvar = self.vae_fc2(h7)
            # x_i = self.vae_fc3(h7)
            # y_i = self.vae_fc4(h7)

        return msk_h1, msk_h2, msk_h3, msk_h4, mu, logvar, x_i, y_i

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, msk_h1, msk_h2, msk_h3, msk_img_feat, z):
        a1 = self.vae_d1(z)
        a2 = self.vae_d1_2(a1)
        a3 = self.vae_d1_3(a2)
        a4 = self.vae_d1_4(a3)

        adain_params = torch.cat((a1, a2, a3, a4), dim=1)
        self.assign_adain_params(adain_params, self)

        # h1 = h1.view(-1, self.ngf *8, 8, 8) 
        # self.assign_adain_params(h1, self)

        h3 = self.adain_layer(msk_img_feat)
        # h2 = self.vae_d2(self.up1(h1)) 
        # h3 = self.vae_d3(self.up2(h2)) 
        # h3 = self.vae_d_SPADE1(h3, msk_img_feat) 
        h4 = self.vae_d4(self.up3(h3))
        h5 = self.vae_d5(self.up4(h4 + msk_h3))
        h6 = self.vae_d6(self.up5(h5 + msk_h2))
        h7 = self.vae_d7(self.up6(h6 + msk_h1))

        return self.vae_d8(h7)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_img_feat, mu, logvar, x_i, y_i = self.encode(x, target)
        if target is not None:
            z = self.reparametrize(mu, logvar)
            # new_y_i = self.reparametrize(y_i, x_i)
            # new_x_i = self.reparametrize(y_i, x_i)
        else:
            z = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_x_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        # adain_params = torch.cat((new_y_i, new_x_i), dim=1)
        # self.assign_adain_params(adain_params, self)
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_img_feat, z)

        return res, mu, logvar, x_i, y_i

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv5


class BoundaryVAEv5(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv5, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = PyrAttnBlock(self.RGB, ndf, 3, 2, 2, False)
        self.ri_e2 = PyrAttnBlock(ndf, ndf * 2, 3, 2, 2, False)
        self.ri_e3 = PyrAttnBlock(ndf * 2, ndf * 4, 3, 2, 2, False)
        self.ri_e4 = PyrAttnBlock(ndf * 4, ndf * 8, 3, 2, 2, False)
        # self.ri_e1 = ConvBlock(self.RGB, ndf, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.ri_e2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu') 

        ### masked image encoder (still use during testing) 
        self.mi_e1 = PyrAttnBlock(input_nc, ndf, 3, 2, 2, True)
        self.mi_e2 = PyrAttnBlock(ndf, ndf * 2, 3, 2, 2, True)
        self.mi_e3 = PyrAttnBlock(ndf * 2, ndf * 4, 3, 2, 2, True)
        self.mi_e4 = PyrAttnBlock(ndf * 4, ndf * 8, 3, 2, 2, True)
        # self.mi_e1 = GatedConvBlock(input_nc, ndf, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.mi_e2 = GatedConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.mi_e3 = GatedConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu') 
        # self.mi_e4 = GatedConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu') 

        ### shared encoder and vae encoder (not use during testing) 
        # self.shrd_e_SPADE1 = SPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')
        self.shrd_e1 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')
        self.shrd_e2 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')

        self.vae_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.vae_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar
        # self.vae_fc3 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # x_i
        # self.vae_fc4 = LinearBlock(ndf *8*8*8, latent_variable_size, norm='batch', acti=None) # y_i

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)

        # self.vae_d1 = LinearBlock(latent_variable_size, latent_variable_size*2, norm=None, acti=None) 
        # self.vae_d1_2 = LinearBlock(latent_variable_size*2, 128*2, norm=None, acti=None) 
        # self.vae_d1_3 = LinearBlock(128*2, 64*2, norm=None, acti=None) 
        # self.vae_d1_4 = LinearBlock(64*2, 32*2, norm=None, acti=None) 

        # self.adain_layer = AdaptiveInstanceNorm2d(latent_variable_size) 

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d2 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d3 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')
        # self.vae_d_SPADE1 = SPADE(ngf * 8, ngf * 8, 3, 1, 1, norm='instance')
        self.vae_d_SPADEResBlk1 = SPADEResnetBlock(ngf * 8, ngf * 8, ngf * 4, 1)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d4 = ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')
        self.vae_d_SPADEResBlk2 = SPADEResnetBlock(ngf * 4, ngf * 4, ngf * 2, 1)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d5 = ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')
        self.vae_d_SPADEResBlk3 = SPADEResnetBlock(ngf * 2, ngf * 2, ngf, 1)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d6 = ConvBlock(ngf * 2, ngf, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')
        self.vae_d_SPADEResBlk4 = SPADEResnetBlock(ngf, ngf, ngf, 1)

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d7 = ConvBlock(ngf, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.vae_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='tanh')

        self.gap = nn.AdaptiveAvgPool2d(1)

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        mu = logvar = x_i = y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32,
                                              device=x.get_device())

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)

            # h5 = self.shrd_e_SPADE1(rl_h4, msk_h4)
            h5 = rl_h4 + msk_h4
            h6 = self.shrd_e1(h5)
            h7 = self.shrd_e2(h6)

            h7 = h7.view(-1, self.ndf * 8 * 8 * 8)
            mu = self.vae_fc1(h7)
            logvar = self.vae_fc2(h7)
            # x_i = self.vae_fc3(h7)
            # y_i = self.vae_fc4(h7)

        return msk_h1, msk_h2, msk_h3, msk_h4, mu, logvar, x_i, y_i

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, msk_h1, msk_h2, msk_h3, msk_img_feat, z):
        # gap = self.gap(msk_img_feat)
        # gap = gap.view(-1, self.latent_variable_size)
        # a = z + gap
        h1 = self.vae_d1(z)
        # a2 = self.vae_d1_2(a1) 
        # a3 = self.vae_d1_3(a2)
        # a4 = self.vae_d1_4(a3) 

        # adain_params = torch.cat((a1, a2, a3, a4), dim=1)
        # self.assign_adain_params(adain_params, self)

        h1 = h1.view(-1, self.ngf * 8, 8, 8)
        # self.assign_adain_params(h1, self)

        # h3 = self.adain_layer(msk_img_feat)
        h2 = self.vae_d2(self.up1(h1))
        h3 = self.vae_d3(self.up2(h2))
        # h3 = self.vae_d_SPADE1(h3, msk_img_feat)
        h3 = self.vae_d_SPADEResBlk1(h3, msk_img_feat)
        h4 = self.vae_d_SPADEResBlk2(self.up3(h3), msk_h3)
        h5 = self.vae_d_SPADEResBlk3(self.up3(h4), msk_h2)
        h6 = self.vae_d_SPADEResBlk4(self.up3(h5), msk_h1)
        h7 = self.vae_d7(self.up6(h6))
        # h4 = self.vae_d4(self.up3(h3)) 
        # h5 = self.vae_d5(self.up4(h4 + msk_h3)) 
        # h6 = self.vae_d6(self.up5(h5 + msk_h2)) 
        # h7 = self.vae_d7(self.up6(h6 + msk_h1)) 

        return self.vae_d8(h7)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_img_feat, mu, logvar, x_i, y_i = self.encode(x, target)
        if target is not None:
            z = self.reparametrize(mu, logvar)
            # new_y_i = self.reparametrize(y_i, x_i)
            # new_x_i = self.reparametrize(y_i, x_i)
        else:
            z = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_x_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        # adain_params = torch.cat((new_y_i, new_x_i), dim=1)
        # self.assign_adain_params(adain_params, self)
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_img_feat, z)

        return res, mu, logvar, x_i, y_i

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define SPADE


class SPADE(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=1, bias=True, pad_type='zero', norm=None,
                 scale_factor=1):
        super(SPADE, self).__init__()
        self.use_bias = bias
        self.nhidden = 128
        self.scale_factor = scale_factor

        # initialize padding 
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

            # initialize normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(output_nc, affine=False)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_nc, affine=False)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(input_nc, self.nhidden, kernel_size, stride),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(self.nhidden, output_nc, kernel_size, stride)
        self.mlp_beta = nn.Conv2d(self.nhidden, output_nc, kernel_size, stride)

        self.down = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    def forward(self, x_featmap, c_featmap):
        # x_featmap: input feature map 
        # c_featmap: conditioned feature map 
        normalized = self.norm(x_featmap)

        if self.scale_factor != 1:
            c_featmap = self.down(c_featmap)
        actv = self.mlp_shared(self.pad(c_featmap))
        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))

        # apply scale and bias 
        out = normalized * (1 + gamma) + beta

        return out

    # Define GatedSPADE


class GatedSPADE(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=1, bias=True, pad_type='zero', norm=None,
                 scale_factor=1):
        super(GatedSPADE, self).__init__()
        self.use_bias = bias
        self.nhidden = 128
        self.scale_factor = scale_factor

        # initialize padding 
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

            # initialize normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(output_nc, affine=False)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_nc, affine=False)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

            # self.mlp_shared = nn.Sequential(
        #     nn.Conv2d(input_nc, self.nhidden, kernel_size, stride), 
        #     nn.ReLU()
        # )
        self.mlp_shared = GatedConvBlock(input_nc, self.nhidden, kernel_size, stride, acti='lrelu')
        self.mlp_gamma = nn.Conv2d(self.nhidden, output_nc, kernel_size, stride)
        self.mlp_beta = nn.Conv2d(self.nhidden, output_nc, kernel_size, stride)

        self.down = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    def forward(self, x_featmap, c_featmap):
        # x_featmap: input feature map 
        # c_featmap: conditioned feature map 
        normalized = self.norm(x_featmap)

        if self.scale_factor != 1:
            c_featmap = self.down(c_featmap)
        actv = self.mlp_shared(self.pad(c_featmap))
        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))

        # apply scale and bias 
        out = normalized * (1 + gamma) + beta

        return out

    # Define the BoundaryVAEv6


class BoundaryVAEv6(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv6, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = PyrAttnBlock(input_nc, ndf, 3, 2, 2, False)
        self.ri_e2 = PyrAttnBlock(ndf, ndf * 2, 3, 2, 2, False)
        self.ri_e3 = PyrAttnBlock(ndf * 2, ndf * 4, 3, 2, 2, False)
        self.ri_e4 = PyrAttnBlock(ndf * 4, ndf * 8, 3, 2, 2, False)
        self.ri_e5 = PyrAttnBlock(ndf * 8, ndf * 8, 3, 2, 2, False)
        self.ri_e6 = PyrAttnBlock(ndf * 8, ndf * 8, 3, 2, 2, False)

        ### masked image encoder (still use during testing) 
        self.mi_e1 = PyrAttnBlock(input_nc, ndf, 3, 2, 2, True)
        self.mi_e2 = PyrAttnBlock(ndf, ndf * 2, 3, 2, 2, True)
        self.mi_e3 = PyrAttnBlock(ndf * 2, ndf * 4, 3, 2, 2, True)
        self.mi_e4 = PyrAttnBlock(ndf * 4, ndf * 8, 3, 2, 2, True)
        self.mi_e5 = PyrAttnBlock(ndf * 8, ndf * 8, 3, 2, 2, True)
        self.mi_e6 = PyrAttnBlock(ndf * 8, ndf * 8, 3, 2, 2, True)

        ### shared encoder and vae encoder (not use during testing) 
        self.ri_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.ri_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        self.mi_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.mi_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)

        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d_SPADEResBlk1 = SPADEResnetBlock(ngf * 8, ngf * 8, ngf * 8, 1)
        # self.vae_d2 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d_SPADEResBlk2 = SPADEResnetBlock(ngf * 8, ngf * 8, ngf * 8, 1)
        # self.vae_d3 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')
        # self.vae_d_SPADE1 = SPADE(ngf * 8, ngf * 8, 3, 1, 1, norm='instance')
        # self.vae_d_SPADEResBlk1 = SPADEResnetBlock(ngf * 8, ngf * 8, ngf * 4, 1)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d_SPADEResBlk3 = SPADEResnetBlock(ngf * 8, ngf * 8, ngf * 8, 1)
        # self.vae_d4 = ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')
        # self.vae_d_SPADEResBlk2 = SPADEResnetBlock(ngf * 4, ngf * 4, ngf * 2, 1)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d5 = ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')
        self.vae_d_SPADEResBlk4 = SPADEResnetBlock(ngf * 4, ngf * 8, ngf * 4, 1)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.vae_d6 = ConvBlock(ngf * 2, ngf, 3, 1, 1, pad_type='replicate', norm='adain', acti='lrelu')
        self.vae_d_SPADEResBlk5 = SPADEResnetBlock(ngf * 2, ngf * 4, ngf * 2, 1)

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d_SPADEResBlk6 = SPADEResnetBlock(ngf, ngf * 2, ngf, 1)

        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.vae_d7 = ConvBlock(ngf, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.vae_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='tanh')

        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        msk_h5 = self.mi_e5(msk_h4)
        msk_h6 = self.mi_e6(msk_h5)

        mu = logvar = x_i = y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32,
                                              device=x.get_device())

        msk_h7 = msk_h6.view(-1, self.ndf * 8 * 8 * 8)
        mu = self.mi_fc1(msk_h7)
        logvar = self.mi_fc2(msk_h7)

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)
            rl_h5 = self.ri_e5(rl_h4)
            rl_h6 = self.ri_e6(rl_h5)

            # h5 = self.shrd_e_SPADE1(rl_h4, msk_h4)
            # h5 = rl_h4 + msk_h4
            # h6 = self.shrd_e1(h5)
            # h7 = self.shrd_e2(h6)

            rl_h7 = rl_h6.view(-1, self.ndf * 8 * 8 * 8)
            # mu = self.vae_fc1(h7)
            # logvar = self.vae_fc2(h7)
            x_i = self.ri_fc1(rl_h7)
            y_i = self.ri_fc1(rl_h7)

        return msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, mu, logvar, x_i, y_i

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def normal_parse_params(self, mu, logvar, min_sigma=1e-3):
        # n = params.shape[0]
        # d = params.shape[1]
        sigma = F.softplus(logvar)
        sigma = sigma.clamp(min=min_sigma)
        distr = Normal(mu, sigma)

        return distr

    def decode(self, msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, z):
        # gap = self.gap(msk_img_feat)
        # gap = gap.view(-1, self.latent_variable_size)
        # a = z + gap
        h1 = self.vae_d1(z)
        # a2 = self.vae_d1_2(a1) 
        # a3 = self.vae_d1_3(a2)
        # a4 = self.vae_d1_4(a3) 

        # adain_params = torch.cat((a1, a2, a3, a4), dim=1)
        # self.assign_adain_params(adain_params, self)

        h1 = h1.view(-1, self.ngf * 8, 8, 8)
        # self.assign_adain_params(h1, self)

        # h3 = self.adain_layer(msk_img_feat)
        h2 = self.vae_d_SPADEResBlk1(h1, msk_h6)
        h3 = self.vae_d_SPADEResBlk2(self.up2(h2), msk_h5)
        h4 = self.vae_d_SPADEResBlk3(self.up3(h3), msk_h4)
        h5 = self.vae_d_SPADEResBlk4(self.up4(h4), msk_h3)
        h6 = self.vae_d_SPADEResBlk5(self.up5(h5), msk_h2)
        h7 = self.vae_d_SPADEResBlk6(self.up6(h6), msk_h1)

        h8 = self.vae_d7(self.up7(h7))
        # h4 = self.vae_d4(self.up3(h3)) 
        # h5 = self.vae_d5(self.up4(h4 + msk_h3)) 
        # h6 = self.vae_d6(self.up5(h5 + msk_h2)) 
        # h7 = self.vae_d7(self.up6(h6 + msk_h1)) 

        return self.vae_d8(h8)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, mu1, logvar1, mu2, logvar2 = self.encode(x, target)
        if target is not None:
            z1 = self.reparametrize(mu1, logvar1)
            z2 = self.reparametrize(mu2, logvar2)
            # new_y_i = self.reparametrize(y_i, x_i)
            # new_x_i = self.reparametrize(y_i, x_i)
        else:
            z1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            z2 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_x_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        # adain_params = torch.cat((new_y_i, new_x_i), dim=1)
        # self.assign_adain_params(adain_params, self)
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, z1)
        # z1 = self.normal_parse_params(mu1, logvar1)
        # z2 = self.normal_parse_params(mu2, logvar2)
        # print(torch.sum(z1))
        # print(torch.sum(z2))

        return res, F.softmax(z1, dim=1), F.softmax(z2, dim=1), mu1, logvar1, mu2, logvar2

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv7


class BoundaryVAEv7(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv7, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        # self.RGB = 3 # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = ConvBlock(input_nc, ndf, 7, 1, 3, pad_type='reflect', norm='instance', acti='lrelu')  # 512x512x32
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 256x256x64
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 128x128x128
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 64x64x256
        self.ri_e5 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 32x32x256
        self.ri_e6 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 16x16x256
        self.ri_e7 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 8x8x256

        ### masked image encoder (still use during testing) 
        self.mi_e1 = ConvBlock(input_nc, ndf, 7, 1, 3, pad_type='reflect', norm='instance', acti='lrelu')  # 512x512x32
        self.mi_e2 = ConvBlock(ndf, ndf * 2, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 256x256x64
        self.mi_e3 = ConvBlock(ndf * 2, ndf * 4, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 128x128x128
        self.mi_e4 = ConvBlock(ndf * 4, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 64x64x256
        self.mi_e5 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 32x32x256
        self.mi_e6 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 16x16x256
        self.mi_e7 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 8x8x256

        ### shared encoder and vae encoder (not use during testing) 
        self.ri_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # mu
        self.ri_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        self.mi_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # mu
        self.mi_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)

        self.nonlocalBlk = NonLocalBlock(latent_variable_size, latent_variable_size, sub_sample=False)
        # 8x8x256 

        self.bpBlk1 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 8x8x(256+256) to 8x8x256

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk2 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 16x16x(256+256) to 16x16x256

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk3 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 32x32x(256+256) to 32x32x256

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk4 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 64x64x(256+256) to 64x64x256

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk5 = BackPrjBlock(ngf * 8 + ngf * 4, ngf * 4)  # 128x128x(256+128) to 128x128x128

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk6 = BackPrjBlock(ngf * 4 + ngf * 2, ngf * 2)  # 256x256x(128+64) to 256x256x64

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk7 = BackPrjBlock(ngf * 2 + ngf, ngf)  # 512x512x(64+32) to 512x512x32

        self.conv_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='sigmoid')

        self.sig = nn.Sigmoid()

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        msk_h5 = self.mi_e5(msk_h4)
        msk_h6 = self.mi_e6(msk_h5)
        msk_h7 = self.mi_e7(msk_h6)

        mu1 = logvar1 = mu2 = logvar2 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32,
                                                    device=x.get_device())

        msk_h8 = msk_h7.view(-1, self.ndf * 8 * 8 * 8)
        mu1 = self.mi_fc1(msk_h8)
        logvar1 = self.mi_fc2(msk_h8)

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)
            rl_h5 = self.ri_e5(rl_h4)
            rl_h6 = self.ri_e6(rl_h5)
            rl_h7 = self.ri_e7(rl_h6)

            rl_h8 = rl_h7.view(-1, self.ndf * 8 * 8 * 8)
            mu2 = self.ri_fc1(rl_h8)
            logvar2 = self.ri_fc2(rl_h8)

        return msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, mu1, logvar1, mu2, logvar2

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def normal_parse_params(self, mu, logvar, min_sigma=1e-3):
        # n = params.shape[0]
        # d = params.shape[1]
        sigma = F.softplus(logvar)
        sigma = sigma.clamp(min=min_sigma)
        distr = Normal(mu, sigma)

        return distr

    def decode(self, msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, z):
        h1 = self.vae_d1(z)
        h1 = h1.view(-1, self.ngf * 8, 8, 8)
        h1 = self.nonlocalBlk(h1)

        h2 = self.bpBlk1(torch.cat((h1, msk_h7), dim=1))
        h3 = self.bpBlk2(torch.cat((self.up1(h2), msk_h6), dim=1))
        h4 = self.bpBlk3(torch.cat((self.up2(h3), msk_h5), dim=1))
        h5 = self.bpBlk4(torch.cat((self.up3(h4), msk_h4), dim=1))
        h6 = self.bpBlk5(torch.cat((self.up4(h5), msk_h3), dim=1))
        h7 = self.bpBlk6(torch.cat((self.up5(h6), msk_h2), dim=1))
        h8 = self.bpBlk7(torch.cat((self.up6(h7), msk_h1), dim=1))

        return self.conv_d8(h8)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, mu1, logvar1, mu2, logvar2 = self.encode(x, target)
        if target is not None:
            z1 = self.reparametrize(mu1, logvar1)
            z2 = self.reparametrize(mu2, logvar2)
        else:
            z1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            z2 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, z2)

        return res, F.softmax(z1, dim=1), F.softmax(z2, dim=1), mu1, logvar1, mu2, logvar2

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv8


class BoundaryVAEv8(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv8, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        # self.RGB = 3 # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = ConvBlock(input_nc, ndf, 7, 1, 3, pad_type='reflect', norm='instance', acti='lrelu')  # 512x512x32
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 256x256x64
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 128x128x128
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 64x64x256
        self.ri_e5 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 32x32x256
        self.ri_e6 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 16x16x256
        self.ri_e7 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 8x8x256

        ### masked image encoder (still use during testing) 
        self.mi_e1 = ConvBlock(input_nc, ndf, 7, 1, 3, pad_type='reflect', norm='instance', acti='lrelu')  # 512x512x32
        self.mi_e2 = ConvBlock(ndf, ndf * 2, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 256x256x64
        self.mi_e3 = ConvBlock(ndf * 2, ndf * 4, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 128x128x128
        self.mi_e4 = ConvBlock(ndf * 4, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 64x64x256
        self.mi_e5 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 32x32x256
        self.mi_e6 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 16x16x256
        self.mi_e7 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 8x8x256

        ### shared encoder and vae encoder (not use during testing) 
        self.gspade1 = GatedSPADE(ndf * 4, ndf * 4, 3, 1, 1, norm='instance')
        self.gspade2 = GatedSPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')

        self.ri_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # mu
        self.ri_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        # self.mi_fc1 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # mu
        # self.mi_fc2 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # logvar

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)

        self.nonlocalBlk = NonLocalBlock(latent_variable_size, latent_variable_size, sub_sample=False)
        # 8x8x256 

        self.bpBlk1 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 8x8x(256+256) to 8x8x256

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk2 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 16x16x(256+256) to 16x16x256

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk3 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 32x32x(256+256) to 32x32x256

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk4 = BackPrjBlock(ngf * 8 * 2, ngf * 8)  # 64x64x(256+256) to 64x64x256

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk5 = BackPrjBlock(ngf * 8 + ngf * 4, ngf * 4)  # 128x128x(256+128) to 128x128x128

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk6 = BackPrjBlock(ngf * 4 + ngf * 2, ngf * 2)  # 256x256x(128+64) to 256x256x64

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bpBlk7 = BackPrjBlock(ngf * 2 + ngf, ngf)  # 512x512x(64+32) to 512x512x32

        self.conv_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti='sigmoid')

        self.sig = nn.Sigmoid()

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)
        msk_h5 = self.mi_e5(msk_h4)
        msk_h6 = self.mi_e6(msk_h5)
        msk_h7 = self.mi_e7(msk_h6)

        mu1 = logvar1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())

        # msk_h8 = msk_h7.view(-1, self.ndf *8*8*8)
        # mu1 = self.mi_fc1(msk_h8)
        # logvar1 = self.mi_fc2(msk_h8)

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(self.gspade1(rl_h3, msk_h3))
            rl_h5 = self.ri_e5(rl_h4)
            rl_h6 = self.ri_e6(rl_h5)
            rl_h7 = self.ri_e7(self.gspade2(rl_h6, msk_h6))

            rl_h8 = rl_h7.view(-1, self.ndf * 8 * 8 * 8)
            mu1 = self.ri_fc1(rl_h8)
            logvar1 = self.ri_fc2(rl_h8)

        return msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, mu1, logvar1

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def normal_parse_params(self, mu, logvar, min_sigma=1e-3):
        # n = params.shape[0]
        # d = params.shape[1]
        sigma = F.softplus(logvar)
        sigma = sigma.clamp(min=min_sigma)
        distr = Normal(mu, sigma)

        return distr

    def decode(self, msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, z):
        h1 = self.vae_d1(z)
        h1 = h1.view(-1, self.ngf * 8, 8, 8)
        h1 = self.nonlocalBlk(h1)

        h2 = self.bpBlk1(torch.cat((h1, msk_h7), dim=1))
        h3 = self.bpBlk2(torch.cat((self.up1(h2), msk_h6), dim=1))
        h4 = self.bpBlk3(torch.cat((self.up2(h3), msk_h5), dim=1))
        h5 = self.bpBlk4(torch.cat((self.up3(h4), msk_h4), dim=1))
        h6 = self.bpBlk5(torch.cat((self.up4(h5), msk_h3), dim=1))
        h7 = self.bpBlk6(torch.cat((self.up5(h6), msk_h2), dim=1))
        h8 = self.bpBlk7(torch.cat((self.up6(h7), msk_h1), dim=1))

        return self.conv_d8(h8)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, mu1, logvar1 = self.encode(x, target)
        if target is not None:
            z1 = self.reparametrize(mu1, logvar1)
            # z2 = self.reparametrize(mu2, logvar2)
        else:
            z1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # z2 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_h4, msk_h5, msk_h6, msk_h7, z1)

        return res, mu1, logvar1

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv9


class BoundaryVAEv9(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv9, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = ConvBlock(self.RGB + 1, ndf, 7, 1, 3, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 512x512x32
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 256x256x64
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 128x128x128
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 64x64x256
        self.ri_e5 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 32x32x256
        self.ri_e6 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 16x16x256
        self.ri_e7 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 8x8x256

        ### shared encoder and vae encoder (not use during testing) 
        # self.gspade1 = GatedSPADE(ndf * 4, ndf * 4, 3, 1, 1, norm='instance')
        # self.gspade2 = GatedSPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')

        self.ri_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.ri_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        # self.mi_fc1 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # mu
        # self.mi_fc2 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # logvar

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)

        self.NonLocalBlk1 = NonLocalBlock(ngf * 8, ngf * 8, sub_sample=False)
        # 8x8x256 

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.up7 = nn.UpsamplingNearest2d(scale_factor=2)

        self.SpadeResBlk1 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 8, 0.015625)
        self.SpadeResBlk2 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 8, 0.03125)
        self.SpadeResBlk3 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 8, 0.0625)
        self.SpadeResBlk4 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 8, 0.125)
        self.SpadeResBlk5 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 4, 0.25)
        self.SpadeResBlk6 = GatedSPADEResnetBlock(input_nc, ngf * 4, ngf * 2, 0.5)
        self.SpadeResBlk7 = GatedSPADEResnetBlock(input_nc, ngf * 2, ngf, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti=None)

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        mu1 = logvar1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)
            rl_h5 = self.ri_e5(rl_h4)
            rl_h6 = self.ri_e6(rl_h5)
            rl_h7 = self.ri_e7(rl_h6)

            rl_h8 = rl_h7.view(-1, self.ndf * 8 * 8 * 8)
            mu1 = self.ri_fc1(rl_h8)
            logvar1 = self.ri_fc2(rl_h8)

        return mu1, logvar1

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def normal_parse_params(self, mu, logvar, min_sigma=1e-3):
        # n = params.shape[0]
        # d = params.shape[1]
        sigma = F.softplus(logvar)
        sigma = sigma.clamp(min=min_sigma)
        distr = Normal(mu, sigma)

        return distr

    def decode(self, x, z):
        h1 = self.vae_d1(z)
        h1 = h1.view(-1, self.ngf * 8, 8, 8)

        h2 = self.SpadeResBlk1(h1, x)
        h2 = self.NonLocalBlk1(h2)
        h3 = self.SpadeResBlk2(self.up1(h2), x)
        h4 = self.SpadeResBlk3(self.up2(h3), x)
        h5 = self.SpadeResBlk4(self.up3(h4), x)
        h6 = self.SpadeResBlk5(self.up4(h5), x)
        h7 = self.SpadeResBlk6(self.up5(h6), x)
        h8 = self.SpadeResBlk7(self.up6(h7), x)

        return self.conv_d8(self.leakyrelu(h8))

    def forward(self, x, target=None):
        mu1, logvar1 = self.encode(x, target)
        if target is not None:
            z1 = self.reparametrize(mu1, logvar1)
        else:
            z1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        res = self.decode(x, z1)

        return res, mu1, logvar1

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv10


class BoundaryVAEv10(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv10, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        # self.ri_e1 = ConvBlock(self.RGB + 1, ndf, 7, 1, 3, pad_type='reflect', norm='instance', acti='lrelu')     # 512x512x32
        self.ri_e1 = ConvBlock(self.RGB + 1, ndf, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 256x256x32
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 3, 2, 1, pad_type='reflect', norm='instance', acti='lrelu')  # 128x128x32
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 64x64x128
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 32x32x256
        self.ri_e5 = ConvBlock(ndf * 8, ndf * 16, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 16x16x512
        self.ri_e6 = ConvBlock(ndf * 16, ndf * 16, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 8x8x512
        self.ri_e7 = ConvBlock(ndf * 16, ndf * 16, 3, 2, 1, pad_type='reflect', norm='instance',
                               acti='lrelu')  # 4x4x512

        ### shared encoder and vae encoder (not use during testing) 
        # self.gspade1 = GatedSPADE(ndf * 4, ndf * 4, 3, 1, 1, norm='instance')
        # self.gspade2 = GatedSPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')

        self.ri_fc1 = LinearBlock(ndf * 16 * 4 * 4, latent_variable_size, norm='batch', acti=None)  # mu
        self.ri_fc2 = LinearBlock(ndf * 16 * 4 * 4, latent_variable_size, norm=None, acti=None)  # logvar

        # self.mi_fc1 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # mu
        # self.mi_fc2 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # logvar

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 16 * 4 * 4, norm=None, acti=None)

        # self.NonLocalBlk1 = NonLocalBlock(ngf *16, ngf *16, sub_sample=False)
        # 8x8x256 

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)

        self.SpadeResBlk1 = GatedSPADEResnetBlock(input_nc, ngf * 16, ngf * 16, 0.015625)
        self.SpadeResBlk2 = GatedSPADEResnetBlock(input_nc, ngf * 16, ngf * 8, 0.03125)
        self.SpadeResBlk3 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 8, 0.0625)
        self.SpadeResBlk4 = GatedSPADEResnetBlock(input_nc, ngf * 8, ngf * 4, 0.125)
        self.SpadeResBlk5 = GatedSPADEResnetBlock(input_nc, ngf * 4, ngf * 2, 0.25)
        self.SpadeResBlk6 = GatedSPADEResnetBlock(input_nc, ngf * 2, ngf, 0.5)
        self.SpadeResBlk7 = GatedSPADEResnetBlock(input_nc, ngf, ngf // 2, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv_d8 = ConvBlock(ngf // 2, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti=None)

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        mu1 = logvar1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())

        if y is not None:
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)
            rl_h5 = self.ri_e5(rl_h4)
            rl_h6 = self.ri_e6(rl_h5)
            rl_h7 = self.ri_e7(rl_h6)

            rl_h8 = rl_h7.view(-1, self.ndf * 16 * 4 * 4)
            mu1 = self.ri_fc1(rl_h8)
            logvar1 = self.ri_fc2(rl_h8)

        return mu1, logvar1

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def normal_parse_params(self, mu, logvar, min_sigma=1e-3):
        # n = params.shape[0]
        # d = params.shape[1]
        sigma = F.softplus(logvar)
        sigma = sigma.clamp(min=min_sigma)
        distr = Normal(mu, sigma)

        return distr

    def decode(self, x, z):
        h1 = self.vae_d1(z)
        h1 = h1.view(-1, self.ngf * 16, 4, 4)

        h2 = self.SpadeResBlk1(self.up1(h1), x)
        # h2 = self.NonLocalBlk1(h2)
        h3 = self.SpadeResBlk2(self.up2(h2), x)
        h4 = self.SpadeResBlk3(self.up3(h3), x)
        h5 = self.SpadeResBlk4(self.up4(h4), x)
        h6 = self.SpadeResBlk5(self.up5(h5), x)
        h7 = self.SpadeResBlk6(self.up6(h6), x)
        h8 = self.SpadeResBlk7(self.up7(h7), x)

        return self.conv_d8(self.leakyrelu(h8))

    def forward(self, x, target=None):
        mu1, logvar1 = self.encode(x, target)
        if target is not None:
            z1 = self.reparametrize(mu1, logvar1)
        else:
            z1 = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        res = self.decode(x, z1)

        return res, mu1, logvar1

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    # Define the BoundaryVAEv20


class BoundaryVAEv20(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, latent_variable_size):
        super(BoundaryVAEv20, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.RGB = 3  # for real image input during training

        ### real image encoder (not use during testing)
        self.ri_e1 = ConvBlock(self.RGB, ndf, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu')
        self.ri_e4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu')
        # self.ri_NonLocalBlk1 = NonLocalBlock(ndf * 8, sub_sample=False)

        ### masked image encoder (still use during testing) 
        self.mi_e1 = ConvBlock(input_nc, ndf, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, norm='instance', acti='lrelu')
        self.mi_e4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, norm='instance', acti='lrelu')
        # self.mi_NonLocalBlk1 = NonLocalBlock(ndf * 8, sub_sample=False)

        ### shared encoder and vae encoder (not use during testing) 
        # self.shrd_e_SPADE1 = SPADE(ndf * 8, ndf * 8, 3, 1, 1, norm='instance')
        self.shrd_e1 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')
        self.shrd_e2 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, norm='instance', acti='relu')

        self.vae_fc1 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm='batch', acti=None)  # mu
        self.vae_fc2 = LinearBlock(ndf * 8 * 8 * 8, latent_variable_size, norm=None, acti=None)  # logvar

        self.mi_down = nn.UpsamplingBilinear2d(scale_factor=0.0625)
        self.mi_gate = GatedConvBlock(input_nc, latent_variable_size, 3, 1, 1, norm='instance')
        self.mi_gap = nn.AdaptiveAvgPool2d(1)
        self.mi_fc1 = LinearBlock(latent_variable_size, latent_variable_size, norm='batch', acti=None)  # mu
        # self.mi_gap = nn.AdaptiveAvgPool2d(1)
        # self.mi_fc1 = LinearBlock(ndf *8, latent_variable_size, norm='batch', acti=None) # mu

        # self.vae_fc3 = LinearBlock(ndf *8*8*8, latent_variable_size, norm=None, acti=None) # x_i
        # self.vae_fc4 = LinearBlock(ndf *8*8*8, latent_variable_size, norm='batch', acti=None) # y_i

        ### vae decoder (still use during testing) 
        self.vae_d1 = LinearBlock(latent_variable_size, ngf * 8 * 8 * 8, norm=None, acti=None)
        # self.vae_d1 = LinearBlock(latent_variable_size, latent_variable_size*2, norm=None, acti=None)

        # self.adain_layer = AdaptiveInstanceNorm2d(latent_variable_size)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.vae_d2 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.vae_d3 = ConvBlock(ngf * 8, ngf * 8, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')
        # self.vae_d_SPADE1 = SPADE(ngf * 8, ngf * 8, 3, 1, 1, norm='instance')

        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.vae_d4 = ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.vae_d5 = ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.vae_d6 = ConvBlock(ngf * 2, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.up6 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.vae_d7 = ConvBlock(ngf, ngf, 3, 1, 1, pad_type='replicate', norm='instance', acti='lrelu')

        self.vae_d8 = ConvBlock(ngf, output_nc, 3, 1, 1, pad_type='replicate', norm=None, acti=None)

    def encode(self, x, y=None):
        # x: masked image 
        # y: real image 
        msk_h1 = self.mi_e1(x)
        msk_h2 = self.mi_e2(msk_h1)
        msk_h3 = self.mi_e3(msk_h2)
        msk_h4 = self.mi_e4(msk_h3)

        # gap_msk_h4 = self.mi_gap(msk_h4)
        # gap_msk_h4 = gap_msk_h4.view(-1, self.ndf *8)
        # z_msk_img = self.mi_fc1(gap_msk_h4)
        a1 = self.mi_gap(self.mi_gate(self.mi_down(x)))
        a1 = a1.view(-1, self.latent_variable_size)
        z_msk_img = self.mi_fc1(a1)

        # msk_h5 = self.mi_NonLocalBlk1(msk_h4)
        mu = logvar = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())

        if y is not None:
            # msk_h5 = self.mi_NonLocalBlk1(msk_h4)
            rl_h1 = self.ri_e1(y)
            rl_h2 = self.ri_e2(rl_h1)
            rl_h3 = self.ri_e3(rl_h2)
            rl_h4 = self.ri_e4(rl_h3)
            # rl_h5 = self.ri_NonLocalBlk1(rl_h4)

            # h5 = self.shrd_e_SPADE1(rl_h4, msk_h4)
            h5 = rl_h4 + msk_h4
            h6 = self.shrd_e1(h5)
            h7 = self.shrd_e2(h6)

            h7 = h7.view(-1, self.ndf * 8 * 8 * 8)
            mu = self.vae_fc1(h7)
            logvar = self.vae_fc2(h7)
            # x_i = self.vae_fc3(h7)
            # y_i = self.vae_fc4(h7)

        return msk_h1, msk_h2, msk_h3, msk_h4, mu, logvar, z_msk_img

    def reparametrize(self, mu, logvar, z_msk_img):
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = z_msk_img
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, msk_h1, msk_h2, msk_h3, msk_img_feat, z):
        h1 = self.vae_d1(z)
        h1 = h1.view(-1, self.ngf * 8, 8, 8)
        # self.assign_adain_params(h1, self)

        # h3 = self.adain_layer(msk_img_feat)
        h2 = self.vae_d2(self.up1(h1))
        h3 = self.vae_d3(self.up2(h2))
        # h3 = self.vae_d_SPADE1(h3, msk_img_feat) 
        h4 = self.vae_d4(self.up3(h3 + msk_img_feat))
        h5 = self.vae_d5(self.up4(h4 + msk_h3))
        h6 = self.vae_d6(self.up5(h5 + msk_h2))
        h7 = self.vae_d7(self.up6(h6 + msk_h1))

        return self.vae_d8(h7)

    def forward(self, x, target=None):
        msk_h1, msk_h2, msk_h3, msk_h4, mu, logvar, z_msk_img = self.encode(x, target)
        if target is not None:
            z = self.reparametrize(mu, logvar, z_msk_img)
            # new_y_i = self.reparametrize(y_i, x_i)
            # new_x_i = self.reparametrize(y_i, x_i)
        else:
            # z = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            z = z_msk_img
            # new_y_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
            # new_x_i = torch.randn(x.size(0), self.latent_variable_size, dtype=torch.float32, device=x.get_device())
        # adain_params = torch.cat((new_y_i, new_x_i), dim=1)
        # self.assign_adain_params(adain_params, self)
        res = self.decode(msk_h1, msk_h2, msk_h3, msk_h4, z)

        return res, mu, logvar

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model 
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model 
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params
