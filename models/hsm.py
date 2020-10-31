from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import pdb
from models.utils import unet
from matplotlib import pyplot as plt

class HSMNet(nn.Module):
    def __init__(self, maxdisp,clean,level=1):
        super(HSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.clean = clean
        self.feature_extraction = unet()
        self.level = level
            
    
        # block 4
        self.decoder6 = decoderBlock(6,32,32,up=True, pool=True)
        if self.level > 2:
            self.decoder5 = decoderBlock(6,32,32,up=False, pool=True)
        else:
            self.decoder5 = decoderBlock(6,32,32,up=True, pool=True)
            if self.level > 1:
                self.decoder4 = decoderBlock(6,32,32, up=False)
            else:
                self.decoder4 = decoderBlock(6,32,32, up=True)
                self.decoder3 = decoderBlock(5,32,32, stride=(2,1,1),up=False, nstride=1)
        # reg
        self.disp_reg8 = disparityregression(self.maxdisp,16)
        self.disp_reg16 = disparityregression(self.maxdisp,16)
        self.disp_reg32 = disparityregression(self.maxdisp,32)
        self.disp_reg64 = disparityregression(self.maxdisp,64)

   

    def feature_vol(self, refimg_fea, targetimg_fea,maxdisp, leftview=True):
        '''
        diff feature volume
        '''
        width = refimg_fea.shape[-1]
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
        for i in range(min(maxdisp, width)):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:]   = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i]   = torch.abs(featb-feata)
        cost = cost.contiguous()
        return cost


    def forward(self, left, right):
        nsample = left.shape[0]
        conv4,conv3,conv2,conv1  = self.feature_extraction(torch.cat([left,right],0))
        conv40,conv30,conv20,conv10  = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample]
        conv41,conv31,conv21,conv11  = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:]

        feat6 = self.feature_vol(conv40, conv41, self.maxdisp//64)
        feat5 = self.feature_vol(conv30, conv31, self.maxdisp//32)
        feat4 = self.feature_vol(conv20, conv21, self.maxdisp//16)
        feat3 = self.feature_vol(conv10, conv11, self.maxdisp//8)

        feat6_2x, cost6 = self.decoder6(feat6)
        feat5 = torch.cat((feat6_2x, feat5),dim=1)

        feat5_2x, cost5 = self.decoder5(feat5)
        if self.level > 2:
            cost3 = F.upsample(cost5, [left.size()[2],left.size()[3]], mode='bilinear')
        else:
            feat4 = torch.cat((feat5_2x, feat4),dim=1)

            feat4_2x, cost4 = self.decoder4(feat4) # 32
            if self.level > 1:
                cost3 = F.upsample((cost4).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            else:
                feat3 = torch.cat((feat4_2x, feat3),dim=1)

                feat3_2x, cost3 = self.decoder3(feat3) # 32
                cost3 = F.upsample(cost3, [left.size()[2],left.size()[3]], mode='bilinear')
        if self.level > 2:
            final_reg = self.disp_reg32
        else:
            final_reg = self.disp_reg8

        if self.training or self.clean==-1:
            pred3 = final_reg(F.softmax(cost3,1)); entropy = pred3  # to save memory
        else:
            pred3,entropy = final_reg(F.softmax(cost3,1),ifent=True)
            pred3[entropy>self.clean] = np.inf

        if self.training:
            cost6 = F.upsample((cost6).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            cost5 = F.upsample((cost5).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            cost4 = F.upsample(cost4, [left.size()[2],left.size()[3]], mode='bilinear')
            pred6 = self.disp_reg16(F.softmax(cost6,1))
            pred5 = self.disp_reg16(F.softmax(cost5,1))
            pred4 = self.disp_reg16(F.softmax(cost4,1))
            stacked = [pred3,pred4,pred5,pred6]   
            return stacked,entropy
        else:
            return pred3,torch.squeeze(entropy)
