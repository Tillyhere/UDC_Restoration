import torch.nn as nn
import torch.nn.functional as F
import torch
from models.model_utils import AdaptiveInstanceNorm, CALayer, PALayer
#from torchsummary import summary
from config import initialise
from sacred import Experiment
from utils.ops import unpixel_shuffle
import numpy as np
from models.cnn_utils import *

#esp+features_from_LRNet
ex = Experiment("LPLRNet_VII")

ex = initialise(ex)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)



class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m
    
    
class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, "kernel size should be odd"
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(
            inc, 1, self.kernel_size, self.kernel_size
        ).contiguous()
        return F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)


class SmoothDilatedResidualAtrousBlock(nn.Module):
    def __init__(self, channel_num, dialation_start: int = 1, group=1, args=None):
        super().__init__()
        self.args = args

        norm = AdaptiveInstanceNorm

        self.norm1 = norm(channel_num // 2)
        self.norm2 = norm(channel_num // 2)
        self.norm4 = norm(channel_num // 2)
        self.norm8 = norm(channel_num // 2)

        self.pre_conv1 = ShareSepConv(2 * dialation_start - 1)
        self.pre_conv2 = ShareSepConv(4 * dialation_start - 1)
        self.pre_conv4 = ShareSepConv(8 * dialation_start - 1)
        self.pre_conv8 = ShareSepConv(16 * dialation_start - 1)

        self.conv1 = nn.Conv2d(
            channel_num,
            channel_num // 2,
            3,
            1,
            padding=dialation_start,
            dilation=dialation_start,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            channel_num,
            channel_num // 2,
            3,
            1,
            padding=2 * dialation_start,
            dilation=2 * dialation_start,
            groups=group,
            bias=False,
        )

        self.conv4 = nn.Conv2d(
            channel_num,
            channel_num // 2,
            3,
            1,
            padding=4 * dialation_start,
            dilation=4 * dialation_start,
            groups=group,
            bias=False,
        )

        self.conv8 = nn.Conv2d(
            channel_num,
            channel_num // 2,
            3,
            1,
            padding=8 * dialation_start,
            dilation=8 * dialation_start,
            groups=group,
            bias=False,
        )

        self.conv = nn.Conv2d(channel_num * 2, channel_num, 3, 1, padding=1, bias=False)

        self.norm = norm(channel_num)
        self.calayer = CALayer(channel_num)
        self.palayer = PALayer(channel_num)

    def forward(self, x):
        y1 = F.leaky_relu(self.norm1(self.conv1(self.pre_conv1(x))), 0.2)
        y2 = F.leaky_relu(self.norm2(self.conv2(self.pre_conv2(x))), 0.2)
        y4 = F.leaky_relu(self.norm4(self.conv4(self.pre_conv4(x))), 0.2)
        y8 = F.leaky_relu(self.norm8(self.conv8(self.pre_conv8(x))), 0.2)

        y = torch.cat((y1, y2, y4, y8), dim=1)
        y = self.norm(self.conv(y))

        y = y + x

        y = self.palayer(self.calayer(y))
        y = y + x

        return F.leaky_relu(y, 0.2)


class ResidualFFABlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1, args=None):
        super().__init__()
        self.args = args

        norm = AdaptiveInstanceNorm

        self.conv1 = nn.Conv2d(
            channel_num,
            channel_num,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            groups=group,
            bias=False,
        )
        self.norm1 = norm(channel_num)
        self.conv2 = nn.Conv2d(
            channel_num,
            channel_num,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            groups=group,
            bias=False,
        )
        self.norm2 = norm(channel_num)

        self.calayer = CALayer(channel_num)
        self.palayer = PALayer(channel_num)

    def forward(self, x):
        y = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        y = y + x
        y = self.norm2(self.conv2(y))

        y = self.palayer(self.calayer(y))
        y = y + x

        return F.leaky_relu(y, 0.2)
class SmoothDilatedResidualAtrousGuidedBlock(nn.Module):
    def __init__(
        self, in_channel, channel_num, dialation_start: int = 1, group=1, args=None
    ):
        super().__init__()
        self.args = args

        norm = AdaptiveInstanceNorm

        self.norm1 = norm(channel_num // 2)
        self.norm2 = norm(channel_num // 2)
        self.norm4 = norm(channel_num // 2)
        self.norm8 = norm(channel_num // 2)

        self.pre_conv1 = ShareSepConv(2 * dialation_start - 1)
        self.pre_conv2 = ShareSepConv(4 * dialation_start - 1)
        self.pre_conv4 = ShareSepConv(8 * dialation_start - 1)
        self.pre_conv8 = ShareSepConv(16 * dialation_start - 1)

        self.conv1 = nn.Conv2d(
            in_channel,
            channel_num // 2,
            3,
            1,
            padding=dialation_start,
            dilation=dialation_start,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channel,
            channel_num // 2,
            3,
            1,
            padding=2 * dialation_start,
            dilation=2 * dialation_start,
            groups=group,
            bias=False,
        )

        self.conv4 = nn.Conv2d(
            in_channel,
            channel_num // 2,
            3,
            1,
            padding=4 * dialation_start,
            dilation=4 * dialation_start,
            groups=group,
            bias=False,
        )

        self.conv8 = nn.Conv2d(
            in_channel,
            channel_num // 2,
            3,
            1,
            padding=8 * dialation_start,
            dilation=8 * dialation_start,
            groups=group,
            bias=False,
        )

        self.conv = nn.Conv2d(channel_num * 2, in_channel, 3, 1, padding=1, bias=False)
        self.norm = norm(in_channel)

    def forward(self, x):
        y1 = F.leaky_relu(self.norm1(self.conv1(self.pre_conv1(x))), 0.2)
        y2 = F.leaky_relu(self.norm2(self.conv2(self.pre_conv2(x))), 0.2)
        y4 = F.leaky_relu(self.norm4(self.conv4(self.pre_conv4(x))), 0.2)
        y8 = F.leaky_relu(self.norm8(self.conv8(self.pre_conv8(x))), 0.2)

        y = torch.cat((y1, y2, y4, y8), dim=1)

        y = self.norm(self.conv(y))

        y = y + x

        return F.leaky_relu(y, 0.2)

    
class EESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'): #down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param g: number of groups to be used in the feature map reduction step.
        '''
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        #assert nIn%k == 0, "Number of input channels ({}) should be divisible by # of branches ({})".format(nIn, k)
        #assert n % k == 0, "Number of output channels ({}) should be divisible by # of branches ({})".format(n, k)
        self.proj_1x1 = CR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        #self.bn = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
            #self.bn.append(nn.BatchNorm2d(n))
        self.conv_1x1_exp = C(nOut, nOut, 1, 1, groups=k)
        self.air_after_cat = AIR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            # We donot combine the branches that have the same effective receptive (3x3 in our case)
            # because there are no holes in those kernels.
            out_k = out_k + output[k - 1]
            #apply batch norm after fusion and then append to the list
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp( # Aggregate the feature maps using point-wise convolution
            self.air_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # as Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class C3block(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          groups=nIn, bias=False),
                AdaptiveInstanceNorm(nIn),
                nn.PReLU(nIn),
                nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          groups=nIn, bias=False),
                AdaptiveInstanceNorm(nIn),
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
    
    
class C3module(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, D_rate=[2,4,8,16]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 4 * n

        self.c1 = C(nIn, n, 1, 1)
        self.d1 = C3block(n, n + n1, 3, 1, D_rate[0])
        self.d2 = C3block(n, n, 3, 1, D_rate[1])
        self.d3 = C3block(n, n, 3, 1, D_rate[2])
        self.d4 = C3block(n, n, 3, 1, D_rate[3])

        self.air = AIR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        combine = torch.cat([d1, d2, d3, d4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.air(combine)
        return output

    
class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output    
    
    
class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        self.air = AIR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.air(combine)
        return output    
    
    

class DilatedParllelResidualBlockB_h(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        #n = int(nOut/4)
        #n1 = nOut - 3*n
        #self.c1 = C(nIn//2, nOut//2, 1, 1)
        self.c1_d = CDilated(nIn//2, nOut//2, 3, 1, 1) # dilation rate of 2^0
        self.c1_r = C(nIn//2,nIn//2,3)
        self.c2_d = CDilated(nIn//4, nOut//4, 3, 1, 2)
        self.c2_r = C(nIn//4,nOut//4,3)
        self.c3_d = CDilated(nIn//8, nOut//8, 3, 1, 4)
        self.c3_r = C(nIn//8,nOut//8,3)
        self.c4_d = CDilated(nIn//16, nOut//16, 3, 1, 8)
        self.c4_r = C(nIn//16, nOut//16,3)
        self.c5_d = CDilated(nIn//16, nOut//16,3, 1, 16)
      
        self.air = AIR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out1, out2= torch.chunk(input, 2, dim=1)
        d1 = self.c1_d(out1)
        r1 = self.c1_r(out2)+out2
        out1, out2 = torch.chunk(r1, 2, dim=1)
        d2 = self.c2_d(out1)
        r2 = self.c2_r(out2)+out2
        out1, out2 = torch.chunk(r2, 2, dim=1)
        d3 = self.c3_d(out1)
        r3 = self.c3_r(out2)+out2
        out1, out2 = torch.chunk(r3, 2 ,dim=1)
        d4 = self.c4_d(out1)
        r4 = self.c4_r(out2)+out2
        d5 = self.c5_d(r4)
        
        '''
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        '''
        #merge
        combine = torch.cat([d1, d2, d3, d4, d5], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.air(combine)
        return output    
    
    
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=2):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        #mypyr = []
        for _ in range(self.num_high):
            #mypyr.append(current)
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        #mypyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class modifiedResBlock(nn.Module):
    def __init__(self, in_features):
        super(modifiedResBlock, self).__init__()
 
        self.proj = nn.Conv2d(in_features, int(in_features//2), 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(in_features//2),int(in_features//2), 3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Conv2d(int(in_features//2), int(in_features//2), 3, padding=1)

    def forward(self, x):
        input = x
        x = self.proj(x)
        x1 = self.conv1(x)
    
        x2 = self.conv2(x1)
        return input + torch.cat((x1,x2),dim=1)

class ESPLRNet(nn.Module):
    def __init__(self, in_c=4, out_c=3, args=None):
        super().__init__()

        self.args = args

        interm_channels = 48
        residual_adds = 4
        #smooth_dialated_block = SmoothDilatedResidualAtrousBlock
        dmodule = DilatedParllelResidualBlockB_h
        residual_block = ResidualFFABlock
        res_block = ResidualBlock
        #rfdb = RFDB
        norm = AdaptiveInstanceNorm

        self.conv1 = nn.Conv2d(in_c, interm_channels, 3, 1, 1, bias=False)
        self.norm1 = norm(interm_channels)
        self.res1_a = res_block(48)
        self.res1_b = res_block(48)
        self.res1_c = res_block(48)
        self.res1_d = res_block(48)
        
        self.res2_a = dmodule(48,48)
        self.res2_b = dmodule(48,48)
        self.res2_c = dmodule(48,48)
        self.res2_d = dmodule(48,48)
        
        self.res3_a = dmodule(48,48)
        self.res3_b = dmodule(48,48)
        self.res3_c = dmodule(48,48)
        self.res3_d = dmodule(48,48)
        
        self.res_final = residual_block(interm_channels, args=args)

        self.gate = nn.Conv2d(
            interm_channels * residual_adds, residual_adds, 3, 1, 1, bias=True
        )

        self.deconv2 = nn.Conv2d(interm_channels, interm_channels, 3, 1, 1)
        #self.norm5 = norm(interm_channels)
        self.deconv1 = nn.Conv2d(interm_channels, out_c, 1)

    def forward(self, x):
        y1 = F.leaky_relu(self.conv1(x), 0.2)

        y = self.res1_a(y1)
        y = self.res1_b(y)
        y = self.res1_c(y)
        y2 = self.res1_d(y)

        y = self.res2_a(y2)
        y = self.res2_b(y)
        y = self.res2_c(y)
        y3 = self.res2_d(y)

        y = self.res3_a(y3)
        y = self.res3_b(y)
        y = self.res3_c(y)
        y = self.res3_d(y)
        y4 = self.res_final(y)

        gates = self.gate(torch.cat((y1, y2, y3, y4), dim=1))
        gated_y = (
            y1 * gates[:, [0], :, :]
            + y2 * gates[:, [1], :, :]
            + y3 * gates[:, [2], :, :]
            + y4 * gates[:, [3], :, :]
        )

        y_1 = self.deconv2(gated_y)
        y= F.leaky_relu(self.norm1(y_1), 0.2)
        y = F.leaky_relu(self.deconv1(y), 0.2)
        y_fused = torch.cat((y1, y2, y3, y4), dim=1)
        return y, y_fused

class OurLRNet(nn.Module):
    def __init__(self, args,radius=1):
        super().__init__()
        #self.pixelshuffle_ratio =2
        self.args = args
        self.lr = ESPLRNet(
            in_c=3 * args.pixelshuffle_ratio ** 2,
            out_c=3 * args.pixelshuffle_ratio ** 2,
            args=args,
        )

    def forward(self, x):
        # Unpixelshuffle
        x_unpixelshuffled = unpixel_shuffle(x, self.args.pixelshuffle_ratio)
       # x_unpixelshuffled = unpixel_shuffle(x, self.pixelshuffle_ratio)

        # Pixelshuffle
        y, y_fused = self.lr(x_unpixelshuffled)
        y_lr = F.pixel_shuffle(
            y, self.args.pixelshuffle_ratio
         #    self.lr(x_unpixelshuffled), self.pixelshuffle_ratio
        )

        return F.tanh(
           y_lr
        ), y_fused
    

class LPLRNet_XIX(nn.Module):
    def __init__(self, args,nrb_low=5, nrb_high=2, num_high=2):
        super(LPLRNet_XIX, self).__init__()

        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        self.esa = ESA(256, nn.Conv2d)
        trans_low = OurLRNet(args=args)
        #trans_high = Trans_high(nrb_high, num_high=num_high)
        self.trans_low = trans_low.cuda()
        #self.trans_high = trans_high.cuda()
        self.conv0 = nn.Conv2d(192, 256, 3, padding=1)
        
        self.conv1_1 =  nn.Conv2d(19,64,3,padding=1)
        self.resblock1_1 = modifiedResBlock(64)
        self.conv1_2 = nn.Conv2d(64, 3, 3,padding=1)
        #self.calayer2 = CALayer(16)
        
        self.conv2_1 = nn.Conv2d(23,32, 3, padding=1)
        #self.resblock2 = ResidualBlock()
        #self.conv2_3 = nn.Conv2d(64,3,3,padding=1)
        self.resblock2_1 = modifiedResBlock(32)
        self.conv2_2 = nn.Conv2d(32, 3, 3, padding=1)
        #self.resblock2_2 = ResidualBlock(64)
        

    def forward(self, real_A_full):

        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
        fake_B_low, y_fused = self.trans_low(pyr_A[-1])
        #pyr_A[-1] = fake_B_low
        fake_B_up = self.lap_pyramid.upsample(fake_B_low)+pyr_A[-2]
        y_fused = self.conv0(y_fused)
        y_fused = self.esa(y_fused)
        y_1_up = F.pixel_shuffle(y_fused, 4)
        y_2_up = F.pixel_shuffle(y_fused, 8)
        #y_3_up = F.pixel_shuffle(y_fused, 16)
        
        high_with_low1 = torch.cat((y_1_up, pyr_A[-2]), dim=1)
        
        #high_with_low3 = torch.cat((y_3_up, pyr_A[-4]), dim=1)
    
        high1_1 = self.conv1_1(high_with_low1)
        high1 = self.resblock1_1(high1_1)
        #high1 = self.resblock1_2(high1)
        #high1 = self.conv1_2(high1)
        #high1 = self.calayer1(high1)
        high1_up = F.pixel_shuffle(high1, 2)
        #high1 = self.conv1_2(high1)
        out1 = self.conv1_2(high1)
        fake_B_up = self.lap_pyramid.upsample(fake_B_up+out1)+pyr_A[-3]
        
        high_with_low2 = torch.cat((high1_up,y_2_up,pyr_A[-3]), dim=1)
        high2_1 = self.conv2_1(high_with_low2)
        high2 = self.resblock2_1(high2_1)
        #high2 = self.resblock2_2(high2)
        #high2_up = F.pixel_shuffle(high2, 2)
        #high2 = self.calayer2(high2)
        #high2 = self.conv2_2(high2)
        out2 = self.conv2_2(high2)
        fake_B_up = fake_B_up+out2
        
    
        return fake_B_up
    
    
@ex.automain
def main(_run):
    from utils.tupperware import tupperware
#   from torchsummary import summary
    from torchscan  import summary
#    from thop import profile
#    from torchstat import stat
#    import torchscan 
#    from ptflops import get_model_complexity_info
    
    
    args = tupperware(_run.config)
    device = args.device
    #model = OurLRNet(args=args).to(args.device)
    #model = SmoothDilatedResidualAtrousBlock(channel_num=48).to("cuda")
    
    model = LPLRNet_XIX(nrb_low=5, nrb_high=3, num_high=2,args=args).to("cuda")    
    dummy_input = torch.randn(1, 3, 1024, 2048,dtype=torch.float).to("cuda")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)
    
    
    
#    from torchstat import stat
    from ptflops import get_model_complexity_info
    args = tupperware(_run.config)
    #model = OurLRNet(args=args).to(args.device)
    #model = SmoothDilatedResidualAtrousBlock(channel_num=48).to("cuda")
    model = LPLRNet_XIX(nrb_low=5, nrb_high=3, num_high=3,args=args).to("cuda")
    #model = OurLRNet(args=args)
    #summary(model,(48,512,512))
    #input = torch.randn(1, 3, 1024, 2048)
    #macs, params = profile(model, inputs=(input,))
    #stat(model, (3, 512, 512))
    flops, params = get_model_complexity_info(model, (3, 1024, 2048), as_strings=True, print_per_layer_stat=True)
    
    

  
