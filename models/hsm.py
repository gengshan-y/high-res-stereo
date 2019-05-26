from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
import pdb
from models.utils import unet
from matplotlib import pyplot as plt

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels, 1)

    target = Variable(target)

    return target

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """

    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

class HSMNet(nn.Module):
    def __init__(self, maxdisp,clean,debug=False,args=None):
        super(HSMNet, self).__init__()
        self.debug = debug
        self.maxdisp = maxdisp
        self.clean = clean
        self.feature_extraction = unet()
        self.args = args
        self.level = args.level
    
        # block 4
        self.decoder6 = decoderBlock(6,32,32,12,up=True,args = args, pool=True)
        #self.decoder6 = decoderBlock(6,32,32,12,up=True,args = args)
        if self.level > 2:
            self.decoder5 = decoderBlock(6,32,32,12, up=False,args = args, pool=True)
        else:
            self.decoder5 = decoderBlock(6,32,32,12, up=True,args = args, pool=True)
            #self.decoder5 = decoderBlock(6,32,32,12, up=True,args = args)
            if self.level > 1:
                self.decoder4 = decoderBlock(6,32,32,12,  up=False,args = args)
            else:
                self.decoder4 = decoderBlock(6,32,32,12,  up=True,args = args)
                self.decoder3 = decoderBlock(5,32,32,12,  stride=(2,1,1),up=False,args = args, nstride=1)
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
        for i in range(maxdisp):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:]   = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i]   = torch.abs(featb-feata)
        cost = cost.contiguous()
        return cost


    def forward(self, left, right,discrete_aux=None):
        nsample = left.shape[0]
        conv4,conv3,conv2,conv1  = self.feature_extraction(torch.cat([left,right],0))
        conv40,conv30,conv20,conv10  = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample]
        conv41,conv31,conv21,conv11  = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:]
       # conv40,conv30,conv20,conv10  = self.feature_extraction(left)
       # conv41,conv31,conv21,conv11  = self.feature_extraction(right)

        feat6 = self.feature_vol(conv40, conv41, self.maxdisp/64)
        feat5 = self.feature_vol(conv30, conv31, self.maxdisp/32)
        feat4 = self.feature_vol(conv20, conv21, self.maxdisp/16)
        feat3 = self.feature_vol(conv10, conv11, self.maxdisp/8)
        feat4r = feat4
        feat3r = feat3

        feat6_2x, cost6, feat6r_2x, cost6r = self.decoder6(feat6, feat6)
        feat5 = torch.cat((feat6_2x, feat5),dim=1)

        feat5_2x, cost5, feat5r, cost5r = self.decoder5(feat5, feat5)
        if self.level > 2:
            #cost3 = F.upsample((cost5).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            cost3 = F.upsample(cost5, [left.size()[2],left.size()[3]], mode='bilinear')
        else:
            feat4 = torch.cat((feat5_2x, feat4),dim=1)

            feat4_2x, cost4, feat4r_2x, cost4r = self.decoder4(feat4, feat4) # 32
            if self.level > 1:
                cost3 = F.upsample((cost4).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            else:
                feat3 = torch.cat((feat4_2x, feat3),dim=1)

                feat3_2x, cost3, feat3r, cost3r = self.decoder3(feat3, feat3) # 32
                # TODO
                cost3 = F.upsample(cost3, [left.size()[2],left.size()[3]], mode='bilinear')
       # #TODO
       # # cost aggregation
       # k=1
       # cost6 = F.upsample(cost6.unsqueeze(1), [cost5.size()[1], left.size()[2]//k//32,left.size()[3]//k//32], mode='trilinear').squeeze(1)
       # cost5 = F.upsample((cost5+cost6).unsqueeze(1), [cost4.size()[1], left.size()[2]//k//16,left.size()[3]//k//16], mode='trilinear').squeeze(1)
       # cost4 = F.upsample(cost4+cost5, [left.size()[2]//k//8,left.size()[3]//k//8], mode='bilinear')
       # cost3 = F.upsample(cost3+cost4, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
        #cost6 = F.upsample(cost6.unsqueeze(1), [cost4.size()[1], left.size()[2]//k,left.size()[3]//k], mode='trilinear').squeeze(1)
        #cost5 = cost6 + F.upsample(cost5.unsqueeze(1), [cost4.size()[1], left.size()[2]//k,left.size()[3]//k], mode='trilinear').squeeze(1)
        #cost4 = cost5 + F.upsample(cost4, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
        #cost3 = cost4 + F.upsample(cost3, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
       # cost4 = F.upsample(cost4, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
       # cost3 = cost4 + F.upsample(cost3, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')

       # pdb.set_trace()
       # from matplotlib import pyplot as plt
       # cost6 = F.upsample(cost6, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
       # cost5 = F.upsample(cost5, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
       # cost4 = F.upsample(cost4, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
       # pred4 = self.disp_reg16(F.softmax(cost4,1))
       # pred4r = pred4
       # pred5 = self.disp_reg32(F.softmax(cost5,1))
       # pred6 = self.disp_reg64(F.softmax(cost6,1))
       # plt.imsave('/data/gengshay/pred6.png',pred6[0])
       # plt.imsave('/data/gengshay/pred5.png',pred5[0])
       # plt.imsave('/data/gengshay/pred4.png',pred4[0])
     #   pred3,entropy = self.disp_reg8(F.softmax(cost3,1),True); pred3r = pred3
     #   pred3[entropy>0.7] = np.inf
        if self.level > 2:
            final_reg = self.disp_reg32
        else:
            final_reg = self.disp_reg8

        if self.clean >0:
            pred3,entropy = final_reg(F.softmax(cost3,1),ifent=True); pred3r = pred3
            pred3[entropy>self.clean] = np.inf
        else:
             # expectation
             pred3 = final_reg(F.softmax(cost3,1)); entropy = pred3; pred3r = pred3
         #   # subpixel map
         #   _,MAP = cost3.max(1)
         #   one_hot = make_one_hot(MAP.unsqueeze(1),cost3.shape[1]).permute(0,2,3,1).contiguous()
         #   wsize=3
         #   weight = Variable(torch.cuda.FloatTensor(cost3.size()[1], cost3.size()[1]).fill_(0))
         #   costN = Variable(torch.cuda.FloatTensor(cost3.size()[0],cost3.size()[1],cost3.size()[2],cost3.size()[3])).fill_(-np.inf)
         #   for w in range(0,cost3.size()[1]):
         #       st = max(0,w-wsize)
         #       weight[w,st:w+wsize+1] = 1
         #   maskN = torch.einsum("abcd,de->abce", (one_hot, weight)).permute(0,3,1,2).byte()
         #   cost3 = torch.where(maskN, cost3, costN)
         #   pred3 = final_reg(F.softmax(cost3,1)); entropy = pred3; pred3r = pred3

        pred3_f = pred3
        pred3r_f = pred3
        pred3r=pred3
        uncertainty = entropy

        if self.training:
            cost6 = F.upsample((cost6).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            cost5 = F.upsample((cost5).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            #cost6 = F.upsample(cost6, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
            #cost5 = F.upsample(cost5, [left.size()[2]//k,left.size()[3]//k], mode='bilinear')
            cost4 = F.upsample(cost4, [left.size()[2],left.size()[3]], mode='bilinear')
            pred4 = self.disp_reg16(F.softmax(cost4,1))
            pred4r = pred4
            #pred5 = self.disp_reg32(F.softmax(cost5,1))
            #pred6 = self.disp_reg64(F.softmax(cost6,1))
            pred5 = self.disp_reg16(F.softmax(cost5,1))
            pred6 = self.disp_reg16(F.softmax(cost6,1))
            if discrete_aux is not None:
                disp_true = discrete_aux[0]
                mask = discrete_aux[1]
                dlabel = (-0.1*torch.abs(self.disp_reg8.disp[:,:,0,0]*self.disp_reg8.divisor - disp_true[mask][:,np.newaxis])).exp()
                dlabel = dlabel / dlabel.sum(-1,keepdim=True)
                stacked = [cost3,cost4,cost5,cost6]
#                loss3 = F.binary_cross_entropy_with_logits(stacked[0].permute(0,2,3,1)[mask],dlabel,size_average=True)*stacked[0].shape[1]
#                loss4 = F.binary_cross_entropy_with_logits(stacked[1].permute(0,2,3,1)[mask],dlabel,size_average=True)*stacked[1].shape[1]
#                loss5 = F.binary_cross_entropy_with_logits(stacked[2].permute(0,2,3,1)[mask],dlabel,size_average=True)*stacked[2].shape[1]
#                loss6 = F.binary_cross_entropy_with_logits(stacked[3].permute(0,2,3,1)[mask],dlabel,size_average=True)*stacked[3].shape[1]
                loss3 = cross_entropy(stacked[0].permute(0,2,3,1)[mask],dlabel,size_average=True)
                loss4 = cross_entropy(stacked[1].permute(0,2,3,1)[mask],dlabel,size_average=True)
                loss5 = cross_entropy(stacked[2].permute(0,2,3,1)[mask],dlabel,size_average=True)
                loss6 = cross_entropy(stacked[3].permute(0,2,3,1)[mask],dlabel,size_average=True)
                stacked = [loss3, loss4, loss5, loss6]
            else:
                stacked = [pred3_f,pred4,pred5,pred6]   
            return stacked, pred3_f,pred4,pred5,pred6,uncertainty
        else:
            return pred3_f,torch.squeeze(uncertainty)
