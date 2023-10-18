
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from SE_weight_module import SEWeightModule
USE_CUDA = True  # gpu
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class groupModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(groupModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        # self.se = SEWeightModule(planes // 4)
        self.se = CPSPPSELayer(planes // 4,planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, -1, feats.shape[2], feats.shape[3])
        # print(' feats',  feats.shape)
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        # print('x1',x1_se.shape)
        # print('x2', x2_se.shape)
        # print('x3', x3_se.shape)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, -1, 1, 1)
        # print('ff', attention_vectors.shape)
        feats_weight = feats * attention_vectors
        feats_weight = self.softmax(feats_weight)
        # print('ff',feats_weight.shape)

        feats_weight = feats*feats_weight#(64,4,16,9,9)
        # print('feats_weight',feats_weight.shape)
        out =feats_weight
        # for i in range(4):
        #     x_se_weight_fp = feats_weight[:, i, :, :, :]
        #     print(x_se_weight_fp.shape)
        #     if i == 0:
        #         out = x_se_weight_fp
        #     else:
        #         out = torch.cat((x_se_weight_fp, out), 1)
        # print('out',out.shape)
        return out


class DPA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(DPA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = groupModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes )
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('out1',out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
       # print('out2',out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        # print('out3', out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
        #print(identity.shape)
        out += identity
        out = self.relu(out)
        return out
class Assio(nn.Module):
    """M_ResBlock module"""
    def __init__(self, band, inter_size):
        super(Assio, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=band, out_channels=64,padding=(0, 0),
                               kernel_size=(1, 1), stride=(1, 1))
        # Dense block
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.PReLU()
        )
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, padding=(0, 0),
                               kernel_size=( 1, 1), stride=(1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=64, padding=(1, 1),
                               kernel_size=(3, 3), stride=(1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=64, padding=(1, 1),
                               kernel_size=(3, 3), stride=(1,  1))
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
       # kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        self.conv5 = nn.Conv2d(in_channels=320, out_channels=64, padding=(0, 0),
                               kernel_size=(1, 1), stride=(1, 1))

        self.batch_norm5 = nn.Sequential(
            nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
    def forward(self, x):
        x1 = self.conv1(x)
        # print('x11', x11.shape)
        x2 = self.batch_norm1(x1)

        x11 = self.conv1(x)
        x11 = self.batch_norm1(x11)
        x2 = torch.cat((x11,x2),dim=1)

        x21 = self.conv2(x2)
        # print('x12', x12.shape)

        x3 = torch.cat((x2, x21), dim=1)
        # print('x13', x13.shape)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        # print('x13', x13.shape)

        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x15', x15.shape)

        # print(x5.shape)
        x6 = self.batch_norm4(x5)
        out = self.conv5(x6)
        out =self.batch_norm5(out)


        return out


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=64):
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features , num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)
        self.bn = nn.BatchNorm1d(64)
        # self-attention graph
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.k = 64
        self.linear_0 = nn.Conv1d(64, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, 64, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        m_batchsize, C, class_num = x.size()
        proj_query = x
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1,
                               keepdim=True)[0].expand_as(energy) - energy
        # print( energy_new.shape)
        attention = self.bn(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        x_glb = self.gamma * out + x
        # x = torch.cat((x_glb, x), dim=1)
        # dynamic_adj=x_glb
        dynamic_adj = self.conv_create_co_mat(x_glb)
        dynamic_adj = torch.sigmoid(dynamic_adj  )
        # idn=x
        # attn = self.linear_0(x)  # b, k, n
        # attn = F.softmax(attn, dim=-1)  # b, k, n
        #
        # attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        # x = self.linear_1(attn)  # b, c, n
        #
        # # x = x.view(b, c, h, w)
        # # x = self.conv2(x)
        # x = x + idn
        # x = F.relu(x)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x, sds=[0, 0, 1]):
        static, dynamic, static_dynamic = sds
        # if static:
        #     out_static = self.forward_static_gcn(x)
        #
        # if dynamic:
        #     dynamic_adj = self.forward_construct_dynamic_graph(x)

        if static_dynamic:
            out_static = self.forward_static_gcn(x)
            x1 = x + out_static  # residual
            dynamic_adj = self.forward_construct_dynamic_graph(x)
            x = self.forward_dynamic_gcn(x1, dynamic_adj)

        return x



class GDPA_LDSICS(nn.Module):
    def  __init__(self,  band, classes ):
        super(GDPA_LDSICS, self).__init__()
        self.name = 'GDPA_LDSICS'
        self.block1= Assio(band,classes)
        self.bn4 = nn.BatchNorm2d(classes)
        self.fc1 = nn.Linear(4096,64)
        self.bn5 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear(self.features_size, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.bn_f1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(1024, 64)
        self.bn_f2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn = nn.BatchNorm1d(16)
        self.DPA = DPA(64, 64)
        self.GCN= DynamicGraphConvolution(64, 64, 64)
        self.conv2d3 = nn.Conv2d(in_channels=36, out_channels=classes, padding=(0, 0),
                                kernel_size=(1, 1), stride=(1, 1))
        self.global_pooling2 = nn.AdaptiveAvgPool2d(1)
        self.global_pooling1 = nn.AdaptiveAvgPool1d(1)
        self.w =nn.Sigmoid()
        self.fc_sam = nn.Conv2d(64, 64, (1, 1), bias=False)
        self.conv_transform = nn.Conv2d(64, 64, (1,1))
        self.ASSIO = ASSOblock(band, 64)
        self.SSD=SSD(band, 64)
        self.SST =SST(band,64)
    def forward_sam(self, x):
        mask = self.fc_sam(x)
        # mask =self.batch_norm1(mask)
        # print('mask', mask.shape)  # 64,64,9,9
        mask = mask.view(mask.size(0), mask.size(1), -1)  # 64,64,81
        mask = torch.sigmoid(mask)  # 64,64,81
        mask = mask.transpose(1, 2)  # 64,81,64
        x = self.conv_transform(x)  # 64,64,9,9
        # x = self.batch_norm1(x)
        x = x.view(x.size(0), x.size(1), -1)  # 64,64,81
        x = torch.matmul(x, mask)  # 64,64,64
        return x


    def forward(self, x):
        (_,c,h,w,b)=x.shape
        #cnn
        xn2=self.SSD(x).squeeze(4)
        #cnn_a
        xn3= self.DPA (xn2)
        xn3=self.global_pooling2(xn3)
        cnn =xn3.squeeze()
        #gcn
        x7 = self.forward_sam(xn2)
        x7 = self.GCN(x7) + x7
        x_7 = x7.view(-1, x7.size(1) * x7.size(2))
        xout = F.leaky_relu(self.fc1(x_7))
        xout = self.bn_f1(xout)

        out = 0.1*xout+0.9*cnn
        # # # out=cnn
        xout = self.fc3(out)
        return xout

class SST(nn.Module):
    def __init__(self,band,batch):
        super(SST,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=band ,out_channels=128,kernel_size=(1,1),padding=(0,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128 ,out_channels=batch,kernel_size=(7,7),padding=(3,3)),
            nn.BatchNorm2d(batch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x1 =self.layer1(x)
        x2=self.layer2(x1)
        return x2

class ASSOblock(nn.Module):
    def __init__(self,band, inter_size):
        super(ASSOblock, self).__init__()

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(1, 1, band), padding=(0, 0, 0),
                      bias=False),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True)
        )

        self.Top_conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=inter_size, kernel_size=(3, 3), padding=(1, 1),bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.Bottom_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0),bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.Bottom_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.Bottom_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.central_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.central_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.end_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.end_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )

        self.end_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=(1, 1), padding=(0, 0),
                      bias=False),
            nn.BatchNorm2d(inter_size),
            nn.ReLU(inplace=True)
        )
        self.end_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=inter_size, out_channels=12, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )

        self.softmax1 = nn.Softmax(dim=1)
        self.sigmiod = nn.Sigmoid()
    def forward(self, x):



        x = self.start(x).squeeze(4)
        xt = self.Top_conv(x)
        # print(xt.shape)
        xt = self.sigmiod(xt)
        xc1 = self.central_conv1(x)
        # print(xc1.shape)
        # xc1 =torch.cat((xt,xc1),dim=1)
        xc1 = xt * xc1
        xc2 = self.central_conv2(xc1)
        xc2 = self.sigmiod(xc2)
        xb1 = self.Bottom_conv1(x)
        xb2 = self.Bottom_conv2(xb1)
        # xb2 =torch.cat((xc2,xb2),dim =1)
        xb2 = xb2 * xc2
        xb3 = self.Bottom_conv3(xb2)
        xb3 = self.sigmiod(xb3)
        xe1 = self.end_conv1(x)
        xe2 = self.end_conv2(xe1)
        xe3 = self.end_conv3(xe2)
        xe3 =xb3*xe3
        xe4 = self.end_conv4(xe3)
        xweight = self.sigmiod(xe4)
        xo = F.relu(x + xe4)
        xo = xo * xweight



        out = self.end_conv(xo)

        return out

class SSD(nn.Module):
    """M_ResBlock module"""
    def __init__(self, band, inter_size):
        super(SSD, self).__init__()
        self.start = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(1, 1, 7), padding=(0, 0, 3),
                      bias=False),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True)
        )

        self.Top_conv = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 1,7), padding=(0, 0,3), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        self.Bottom_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 1,7), padding=(0, 0,3), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        self.Bottom_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1,7), padding=(0, 0,3),
                      bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        self.Bottom_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=24, kernel_size=(1, 1,7), padding=(0, 0,3),
                      bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        self.central_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 1,7), padding=(0, 0,3),
                      bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        self.central_conv2 = nn.Sequential(
            nn.Conv3d(in_channels= 48, out_channels=24, kernel_size=(1, 1,7), padding=(0, 0,3),
                      bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        self.end_conv = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=inter_size, kernel_size=(1, 1,band), padding=(0, 0,0), bias=False),
            nn.BatchNorm3d(inter_size),
            nn.ReLU(inplace=True)
        )
        self.softmax1 = nn.Softmax(dim=1)
        self.sigmiod = nn.Sigmoid()
    def forward(self, x):
        x = self.start(x)
        xt = self.Top_conv(x)
        xc1 = self.central_conv1(x)
        xc1 =torch.cat((xt,xc1),dim=1)
        xc2 = self.central_conv2(xc1)
        xb1 = self.Bottom_conv1(x)
        xb2 = self.Bottom_conv2(xb1)
        xb2 =torch.cat((xc2,xb2),dim =1)
        xb3 = self.Bottom_conv3(xb2)
        xo =self.end_conv(xb3)

        out = xo

        return out


class CPSPPSELayer(nn.Module):
    def __init__(self, in_channel, channel, reduction=21):
        super(CPSPPSELayer, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        # self.avg_pool8 = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel * 21 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 21 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x) if hasattr(self, 'conv1') else x
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        # y4 = self.avg_pool8(x).view(b, 64 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b, out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        # print('y',y.shape)
        return y