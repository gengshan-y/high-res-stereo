import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from torch.autograd import Variable


class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.inplanes = 32

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)

        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, None,  fusion_mode='sum', model_name='icnet')
        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)

        self.proj6 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=32,padding=0,stride=1,bias=False)
        self.proj5 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=16, padding=0,stride=1,bias=False)
        self.proj4 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=16, padding=0,stride=1,bias=False)
        self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=16, padding=0,stride=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight, 0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#                m.running_mean.data.fill_(0)
#                m.running_var.data.fill_(1)
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        conv3 = self.res_block3(pool1)
        conv4 = self.res_block5(conv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        concat5 = torch.cat((conv5,self.upconv6(conv6)),dim=1)
        conv5 = self.iconv5(concat5) 

        concat4 = torch.cat((conv4,self.upconv5(conv5)),dim=1)
        conv4 = self.iconv4(concat4) 

        concat3 = torch.cat((conv3,self.upconv4(conv4)),dim=1)
        conv3 = self.iconv3(concat3) 

        proj6 = self.proj6(conv6)
        proj5 = self.proj5(conv5)
        proj4 = self.proj4(conv4)
        proj3 = self.proj3(conv3)
        return proj6,proj5,proj4,proj3



class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs



class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1):
        super(residualBlock, self).__init__()

        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, bias=False,dilation=dilation)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, with_bn=with_bn))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode
    
    #@profile
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        if self.pool_sizes is None: 
            for pool_size in np.linspace(1,min(h,w)//2,4,dtype=int):
                k_sizes.append((int(h/pool_size), int(w/pool_size)))
                strides.append((int(h/pool_size), int(w/pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0],self.pool_sizes[0]),(self.pool_sizes[1],self.pool_sizes[1]) ,(self.pool_sizes[2],self.pool_sizes[2]) ,(self.pool_sizes[3],self.pool_sizes[3])]
            strides = k_sizes

        if self.fusion_mode == 'cat': # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else: # icnet: element-wise sum (including x)
            pp_sum = x

            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                pp_sum = pp_sum + 0.25*out
            pp_sum = F.relu(pp_sum/2.,inplace=True)

            return pp_sum

