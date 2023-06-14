import torch
import math
from mmcv.ops import DeformConv2dPack as DCN
import torch.nn.functional as F


class FeatureAlign_FSM_Block1(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_FSM_Block1, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_L2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.relu = torch.nn.ReLU(inplace=True)

       
    def forward(self, rgb, depth):

        rgb_att = self.lateral_conv(rgb)  # 0~1 * feats
        offset = self.offset(torch.cat([rgb_att*2, depth], dim=1))  # concat for offset by compute the dif
        depth_align = self.relu(self.dcpack_L2([depth, offset]) )  # [feat, offset]

   
        return depth_align+rgb_att

class FeatureAlign_FSM_Block2(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_FSM_Block2, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")

    def forward(self, rgb, depth):

   
        rgb_att0 = self.lateral_conv0(rgb[0])
        offset0 = self.offsetGen0(torch.cat([rgb_att0*2, depth[0]], dim=1))  # concat for offset
        depth_align0 = self.relu(self.dcpack_0([depth[0], offset0])) 
      

        rgb_att1 = self.lateral_conv1(rgb[1])
        offset1 = self.offsetGen1(torch.cat([rgb_att1*2, depth[1]], dim=1))
        depth_align1 = self.relu(self.dcpack_1([depth[1], offset1]))
      
 
        return depth_align0+rgb_att0 , depth_align1+rgb_att1

class FeatureAlign_FSM_Block3(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_FSM_Block3, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv2 = FeatureSelectionModule(in_nc, out_nc, norm="")

    def forward(self, rgb, depth):

   
        rgb_att0 = self.lateral_conv0(rgb[0])
        offset0 = self.offsetGen0(torch.cat([rgb_att0*2, depth[0]], dim=1))  # concat for offset
        depth_align0 = self.relu(self.dcpack_0([depth[0], offset0])) 
      

        rgb_att1 = self.lateral_conv1(rgb[1])
        offset1 = self.offsetGen1(torch.cat([rgb_att1*2, depth[1]], dim=1))
        depth_align1 = self.relu(self.dcpack_1([depth[1], offset1]))
      
        rgb_att2 = self.lateral_conv2(rgb[2])
        offset2 = self.offsetGen2(torch.cat([rgb_att2*2, depth[2]], dim=1))
        depth_align2 = self.relu(self.dcpack_2([depth[2], offset2]))

        return depth_align0+rgb_att0 , depth_align1+rgb_att1, depth_align2+rgb_att2

class FeatureAlign_Block2_c2f(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_Block2_c2f, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_block(out_nc)
        self.fusion2 = Fusion_block(out_nc)
        self.fuse_conv = torch.nn.Conv2d(out_nc*2, out_nc, kernel_size=1, stride=1, padding=0)
       

    def forward(self, depth, rgb):

        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))
        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([rgb[0], offset0])) 
        feat_fuse1 = self.fusion1(rgb[0], feat_D0)

        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([rgb[1], offset1]))

        feat_D0_up = F.interpolate(feat_D0, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D1 = self.fuse_conv(torch.cat((feat_D1, feat_D0_up), 1))

        feat_fuse2 = self.fusion2(rgb[1], feat_D1)
      
 
        return feat_fuse1, feat_fuse2

class FeatureAlign_Block3(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_Block3, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_block(out_nc)
        self.fusion2 = Fusion_block(out_nc)
        self.fusion3 = Fusion_block(out_nc)
       

    def forward(self, depth, rgb):

        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))
        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([rgb[0], offset0])) 
        feat_fuse1 = self.fusion1(rgb[0], feat_D0)

        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([rgb[1], offset1]))
        feat_fuse2 = self.fusion2(rgb[1], feat_D1)

        offset2 = self.offsetGen2(torch.cat([rgb[2], depth[2]], dim=1))
        feat_D2 = self.relu(self.dcpack_2([rgb[2], offset2]))
        feat_fuse3 = self.fusion3(rgb[2], feat_D2)      
 
        return feat_fuse1, feat_fuse2, feat_fuse3

class FeatureAlign_Block3_c2f(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_Block3_c2f, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)


        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_block(out_nc)
        self.fusion2 = Fusion_block(out_nc)
        self.fusion3 = Fusion_block(out_nc)
        self.fuse_conv1 = torch.nn.Conv2d(out_nc*2, out_nc, kernel_size=1, stride=1, padding=0)
        self.fuse_conv2 = torch.nn.Conv2d(out_nc*2, out_nc, kernel_size=1, stride=1, padding=0)

       

    def forward(self, depth, rgb):

        h, w = int(rgb[2].size(2)), int(rgb[2].size(3))
        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([rgb[0], offset0])) 
        feat_fuse1 = self.fusion1(rgb[0], feat_D0)
#############
        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([rgb[1], offset1]))

        feat_D0_up = F.interpolate(feat_D0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)

        feat_D1 = self.fuse_conv1(torch.cat((feat_D1, feat_D0_up), 1))

        feat_fuse2 = self.fusion2(rgb[1], feat_D1)
###############
        offset2 = self.offsetGen2(torch.cat([rgb[2], depth[2]], dim=1))
        feat_D2 = self.relu(self.dcpack_2([rgb[2], offset2]))

        feat_D1_up = F.interpolate(feat_D1, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D2 = self.fuse_conv2(torch.cat((feat_D2, feat_D1_up), 1))

        feat_fuse3 = self.fusion3(rgb[2], feat_D2)      
 
        return feat_fuse1, feat_fuse2, feat_fuse3


class FeatureAlign_CA_Block1(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_CA_Block1, self).__init__()
        # self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_L2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion = Fusion_Cross_Atten_block(out_nc)
       
    def forward(self, feat_l, feat_s):#(rgb, depth)

        # feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_l*2, feat_s], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_s, offset]) )  # [feat, offset]

        feat_fuse = self.fusion(feat_l, feat_align)
   
        return feat_fuse

class FeatureAlign_CA_Block2(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_Block2, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_Cross_Atten_block(out_nc)
        self.fusion2 = Fusion_Cross_Atten_block(out_nc)
       

    def forward(self, rgb, depth):

   
        offset0 = self.offsetGen0(torch.cat([rgb[0]*2, depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0])) 
        feat_fuse1 = self.fusion1(rgb[0], feat_D0)

        offset1 = self.offsetGen1(torch.cat([rgb[1]*2, depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]))
        feat_fuse2 = self.fusion2(rgb[1], feat_D1)
      
 
        return feat_fuse1, feat_fuse2

class FeatureAlign_CA_Block3(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_Block3, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_Cross_Atten_block(out_nc)
        self.fusion2 = Fusion_Cross_Atten_block(out_nc)
        self.fusion3 = Fusion_Cross_Atten_block(out_nc)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))
        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0]*2, depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0])) 
        feat_fuse1 = self.fusion1(depth[0], feat_D0)

        offset1 = self.offsetGen1(torch.cat([rgb[1]*2, depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]))
        feat_fuse2 = self.fusion2(rgb[1], feat_D1)

        offset2 = self.offsetGen2(torch.cat([rgb[2]*2, depth[2]], dim=1))
        feat_D2 = self.relu(self.dcpack_2([depth[2], offset2]))
        feat_fuse3 = self.fusion3(rgb[2], feat_D2)      
 
        return feat_fuse1, feat_fuse2, feat_fuse3

class Fusion_block(torch.nn.Module):
    def __init__(self, input_dim):
        super(Fusion_block, self).__init__()
        self.conv_f = ConvBlock(2*input_dim, input_dim, 1, 1, 0, activation='prelu', norm=None)
        self.act = torch.nn.Softmax(dim=1)
        self.down = torch.nn.MaxPool2d(8,8)
        self.conv_fup = DeconvBlock(input_dim, input_dim, 12, 8, 2, activation='prelu', norm=None)

    def forward(self, depth, color):      
        
        fuse00 = torch.cat((depth, color), 1)
        fuse0 = self.conv_f(fuse00)
        f0 = self.down(fuse0)


        b, c, h, w = f0.shape

        f_Q = f0.view(b, c, -1)
        f_K = f0.view(b, c, -1).permute(0, 2, 1).contiguous()
        f_V = f0.view(b, c, -1).permute(0, 2, 1).contiguous()
        mut_f_Q_K = self.act(torch.matmul(f_K, f_Q))
        att_f = torch.matmul(mut_f_Q_K, f_V).permute(0, 2, 1).contiguous().view(b, c, h, w)
        att_f = self.conv_fup(att_f)
  
        # return fuse0+0.5*att_f
        return fuse0+1.5*att_f
        # return fuse0+15*att_f

class Fusion_Cross_Atten_block(torch.nn.Module):
    def __init__(self, input_dim):
        super(Fusion_Cross_Atten_block, self).__init__()
        # self.conv_f = ConvBlock(2*input_dim, input_dim, 1, 1, 0, activation='prelu', norm=None)
        # self.act = torch.nn.Softmax(dim=1)
        self.down = torch.nn.MaxPool2d(8,8)
        self.conv_up_d = DeconvBlock(input_dim, input_dim, 12, 8, 2, activation='prelu', norm=None)
        self.conv_up_c = DeconvBlock(input_dim, input_dim, 12, 8, 2, activation='prelu', norm=None)

        self.d_Q = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.d_K = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.d_V = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)

        self.c_Q = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.c_K = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.c_V = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)

        self.act = torch.nn.Softmax(dim=1)

        self.fuse = ConvBlock(2*input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, depth, color):


        d = self.down(depth)
        c = self.down(color)

        d_Q = self.d_Q(d)
        d_K = self.d_K(d)
        d_V = self.d_V(d)

        c_Q = self.c_Q(c)
        c_K = self.c_K(c)
        c_V = self.c_V(c)

        b, c, h, w = d_Q.shape

        v_d_Q = d_Q.view(b, c, -1)
        v_d_K = d_K.view(b, c, -1).permute(0, 2, 1).contiguous()
        v_d_V = d_V.view(b, c, -1).permute(0, 2, 1).contiguous()

        v_c_Q = c_Q.view(b, c, -1)
        v_c_K = c_K.view(b, c, -1).permute(0, 2, 1).contiguous()
        v_c_V = c_V.view(b, c, -1).permute(0, 2, 1).contiguous()

        mut_d_Q_K = self.act(torch.matmul(v_c_K, v_d_Q))
        att_c = torch.matmul(mut_d_Q_K, v_c_V).permute(0, 2, 1).contiguous().view(b, c, h, w)
        att_c = self.conv_up_c(att_c)

        mut_c_Q_K = self.act(torch.matmul(v_d_K, v_c_Q))
        att_d = torch.matmul(mut_c_Q_K, v_d_V).permute(0, 2, 1).contiguous().view(b, c, h, w)
        att_d = self.conv_up_d(att_d)

        fuse_d = depth+1.5*att_d
        fuse_c = color+1.5*att_c

        fuse_d_c = torch.cat((fuse_d, fuse_c), 1)
        fuse = self.fuse(fuse_d_c)
  
        return fuse


class Cross_Attention_Block(torch.nn.Module):
    def __init__(self, num_filter):
        super(Cross_Attention_Block, self).__init__()

        # self.act = torch.nn.Sigmoid()
        self.m_Q = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.m_K = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.m_V = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')

        self.e_Q = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.e_K = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.e_V = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')

        self.act = torch.nn.Softmax(dim=1)

    def forward(self, m, e):
        m_Q = self.m_Q(m)
        m_K = self.m_K(m)
        m_V = self.m_V(m)

        e_Q = self.e_Q(e)
        e_K = self.e_K(e)
        e_V = self.e_V(e)

        b, c, h, w = m_Q.shape

        v_m_Q = m_Q.view(b, c, -1)
        v_m_K = m_K.view(b, c, -1).permute(0, 2, 1).contiguous()
        v_m_V = m_V.view(b, c, -1).permute(0, 2, 1).contiguous()

        v_e_Q = e_Q.view(b, c, -1)
        v_e_K = e_K.view(b, c, -1).permute(0, 2, 1).contiguous()
        v_e_V = e_V.view(b, c, -1).permute(0, 2, 1).contiguous()

        mut_m_Q_K = self.act(torch.matmul(v_e_K, v_m_Q))
        att_e = torch.matmul(mut_m_Q_K, v_e_V).permute(0, 2, 1).contiguous().view(b, c, h, w)

        mut_e_Q_K = self.act(torch.matmul(v_m_K, v_e_Q))
        att_m = torch.matmul(mut_e_Q_K, v_m_V).permute(0, 2, 1).contiguous().view(b, c, h, w)

        # print(att_m.shape, 'mmmmmmmmmmmmmmmmmmm')

        return  att_m + m_V,  att_e + e_V



class FeatureAlign_no_fusion_Block1(torch.nn.Module):  # Alignment
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_no_fusion_Block1, self).__init__()
        # self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_L2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion = Fusion_block(out_nc)
       
    def forward(self, feat_l, feat_s):

        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        # feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_l*2, feat_up], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]) )  # [feat, offset]

        #feat_fuse = self.fusion(feat_l, feat_align)
   
        return feat_l+feat_align

class FeatureAlign_no_fusion_Block2(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_no_fusion_Block2, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_block(out_nc)
        self.fusion2 = Fusion_block(out_nc)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))
 
        offset0 = self.offsetGen0(torch.cat([rgb[0]*2, depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0])) 

        offset1 = self.offsetGen1(torch.cat([rgb[1]*2, depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]))
  
        return rgb[0]+feat_D0, rgb[1]+feat_D1

class FeatureAlign_no_fusion_Block3(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_no_fusion_Block3, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.relu = torch.nn.ReLU(inplace=True)

        self.fusion1 = Fusion_block(out_nc)
        self.fusion2 = Fusion_block(out_nc)
        self.fusion3 = Fusion_block(out_nc)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))
        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0]*2, depth[0]], dim=1))  # concat for offset
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0])) 
        #feat_fuse1 = self.fusion1(rgb[0], feat_D0)

        offset1 = self.offsetGen1(torch.cat([rgb[1]*2, depth[1]], dim=1))
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]))
        #feat_fuse2 = self.fusion2(rgb[1], feat_D1)

        offset2 = self.offsetGen2(torch.cat([rgb[2]*2, depth[2]], dim=1))
        feat_D2 = self.relu(self.dcpack_2([depth[2], offset2]))
        #feat_fuse3 = self.fusion3(rgb[2], feat_D2)      
 
        return rgb[0]+feat_D0, rgb[1]+feat_D1, rgb[2]+feat_D2

class FeatureAlign_CA_c2f_Block3(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_c2f_Block3, self).__init__()
        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv2 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        
        self.c2f0 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        self.c2f1 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        self.c2f2 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        
        self.fusion0 = Fusion_Cross_Atten_block(out_nc)
        self.fusion1 = Fusion_Cross_Atten_block(out_nc)
        self.fusion2 = Fusion_Cross_Atten_block(out_nc)
       
        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[2].size(2)), int(rgb[2].size(3))

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]*2], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0]) ) 
        # feat_align0 = self.c2f0 (feat_D0)
        feat_align0_up =  F.interpolate(feat_D0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)

        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]*2, offset0_up], dim=1))
        offset1_up = F.interpolate(offset1, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]) )
        feat_align1 =self.c2f1 (feat_align0_up +feat_D1)
        feat_align1_up =  F.interpolate(feat_align1, size=(int(h), int(w)), mode='bilinear', align_corners=True)

        offset2 = self.offsetGen2(torch.cat([rgb[2], depth[2]*2, offset1_up], dim=1))
        feat_D2 = self.relu(self.dcpack_2([depth[2], offset2]) )
        feat_align2 =self.c2f2 (feat_align1_up +feat_D2)
        
        feat_fuse0 = self.fusion0(rgb[0], feat_D0)    
        feat_fuse1 = self.fusion1(rgb[1], feat_align1)    
        feat_fuse2 = self.fusion2(rgb[2], feat_align2)    
        
        return feat_fuse0, feat_fuse1, feat_fuse2

class FeatureAlign_CA_c2f_Block2(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_c2f_Block2, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        # self.offsetGen2 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        # self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        
        self.c2f0 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        self.c2f1 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
    
        self.fusion0 = Fusion_Cross_Atten_block(out_nc)
        self.fusion1 = Fusion_Cross_Atten_block(out_nc)
        # self.fusion2 = Fusion_Cross_Atten_block(out_nc)
       
        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):
    
        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]*2], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0]) ) 
        # feat_align0 = self.f0 (feat_D0)
        feat_align0_up =  F.interpolate(feat_D0, size=(int(h), int(w)), mode='bilinear', align_corners=True)

        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]*2, offset0_up], dim=1))
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]) )
        feat_align1 =self.c2f1 (feat_align0_up +feat_D1)
       

        feat_fuse0 = self.fusion0(rgb[0], feat_D0)    
        feat_fuse1 = self.fusion1(rgb[1], feat_align1)    
  
        return feat_fuse0, feat_fuse1

class FeatureAlign_CA_c2f_Block1(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_c2f_Block1, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        # self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        # self.offsetGen2 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        # self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        # self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        self.fusion0 = Fusion_Cross_Atten_block(out_nc)
        # self.fusion1 = Fusion_Cross_Atten_block(out_nc)
        # self.fusion2 = Fusion_Cross_Atten_block(out_nc)
       
        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):

    

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb, depth*2], dim=1))  # concat for offset
        # offset0_up = F.interpolate(offset0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth, offset0]) ) 
        # feat_align0 = self.f0 (feat_D0)
        # feat_align0_up =  F.interpolate(feat_align0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)

        feat_fuse0 = self.fusion0(rgb, feat_D0)    
        
  
        return feat_fuse0




















class Basis_Projection_block(torch.nn.Module):
    def __init__(self):
        super(Basis_Projection_block, self).__init__()
        self.G_depth = torch.nn.AdaptiveAvgPool2d(1)
        self.G_rgb = torch.nn.AdaptiveAvgPool2d(1)


    def forward(self, color, depth):
        if(depth.shape[:2]!=color.shape[:2]):
            raise ()
        
        b, c, _, _ = depth.shape##(b=1 c=1 h=w =128)
        B_depth = self.G_depth(depth).view(b, c, -1)
        B_rgb = self.G_rgb(color).view(b, c, -1)
        Norm_B_depth = torch.div(B_depth, torch.norm(B_depth))
        Norm_B_rgb = torch.div(B_rgb, torch.norm(B_rgb))
        
        out_rgb = torch.mul(Norm_B_rgb.reshape(b, c, 1, 1), depth)
        out_rgb = torch.mul(B_rgb.reshape(b, c, 1, 1), out_rgb)

        out_depth = torch.mul(Norm_B_depth.reshape(b, c, 1, 1), color)
        out_depth = torch.mul(B_depth.reshape(b, c, 1, 1), out_depth)
        return out_rgb, out_depth


class FeatureSelectionModule(torch.nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = torch.nn.Conv2d(in_chan, in_chan, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat

class C2F_AlignTriple(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(C2F_AlignTriple, self).__init__()
        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv2 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        # self.f0 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        # self.f1 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        # self.f2 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, depth, rgb):

        h, w = int(rgb[2].size(2)), int(rgb[2].size(3))

        feat_arm0 = self.lateral_conv0(rgb[0])  #由小到大对RGB加注意力
        feat_arm1 = self.lateral_conv1(rgb[1])
        feat_arm2 = self.lateral_conv2(rgb[2])

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([feat_arm0, depth[0]], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([feat_arm0, offset0]) ) 
        # feat_align0 = self.f0 (feat_D0)
        # feat_align0_up =  F.interpolate(feat_align0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)

        offset1 = self.offsetGen1(torch.cat([feat_arm1, depth[1], 2*offset0_up], dim=1))
        offset1_up = F.interpolate(offset1, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D1 = self.relu(self.dcpack_1([feat_arm1, offset1]) )
        # feat_align1 =self.f1 (feat_align0_up +feat_D1)
        # feat_align1_up =  F.interpolate(feat_align1, size=(int(h), int(w)), mode='bilinear', align_corners=True)

        offset2 = self.offsetGen2(torch.cat([feat_arm2, depth[2],2*offset1_up], dim=1))
        feat_D2 = self.relu(self.dcpack_2([feat_arm2, offset2]) )
        # feat_align2 =self.f1 (feat_align1_up +feat_D2)
      
        return feat_D0 +rgb[0], feat_D1 +rgb[1], feat_D2 +rgb[2]

class C2F_AlignDouble(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(C2F_AlignDouble, self).__init__()
        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)

        # self.f0 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)
        # self.f1 = ResnetBlock(out_nc ,out_nc, 3, 1, 1, activation='prelu', norm=None)

        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, depth, rgb):

        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))

        feat_arm0 = self.lateral_conv0(rgb[0])  #由小到大对RGB加注意力
        feat_arm1 = self.lateral_conv1(rgb[1])


        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([feat_arm0, depth[0]], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([feat_arm0, offset0]) ) 
        # feat_align0 = self.f0 (feat_D0)
        # feat_align0_up =  F.interpolate(feat_align0, size=(int(h), int(w)), mode='bilinear', align_corners=True)

        offset1 = self.offsetGen1(torch.cat([feat_arm1, depth[1],offset0_up*2], dim=1))
        feat_D1 = self.relu(self.dcpack_1([feat_arm1, offset1]) )
        # feat_align1 =self.f1 (feat_align0_up +feat_D1)
      
 
        return feat_D0 + rgb[0], feat_D1 + rgb[1]


class FeatureAlign_V2(torch.nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_L2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.relu = torch.nn.ReLU(inplace=True)
       
    def forward(self, feat_l, feat_s):

        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm*2, feat_up], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]) )  # [feat, offset]
   
        return feat_align + feat_arm

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

# class AvgpoolingBlock(torch.nn.Module):
#     def __init__(self, kernel_size=3, stride=1, padding=1, dilation, return_indices, bias=True, ceil_mode, activation='prelu', norm=None):
#         super(AvgpoolingBlock, self).__init__()
#         self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, dilation=1, return_indices=False, ceil_mode=False)
#         #self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.avgpool(x))
#         else:
#             out = self.avgpool(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out

# class MaxpoolingBlock(torch.nn.Module):
#     def __init__(self, kernel_size=4, stride=4, padding=0, bias=True, activation='prelu', norm=None):
#         super(MaxpoolingBlock, self).__init__()
#         self.avgpool = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, return_indices=False, ceil_mode=False)
#         #self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.avgpool(x))
#         else:
#             out = self.avgpool(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out

class DilaConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=2, bias=True, activation='prelu', norm=None):
        super(DilaConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding,  dilation, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, out_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, out_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(out_filter, out_filter, kernel_size, stride, padding, bias=bias)
        self.conv3 = torch.nn.Conv2d(out_filter, out_filter, kernel_size, stride, padding, bias=bias)
        self.conv4 = torch.nn.Conv2d(out_filter, out_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(out_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(out_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        #residual = x
        if self.norm is not None:
            out1 = self.bn(self.conv1(x))
        else:
            out1 = self.conv1(x)

        if self.activation is not None:
            out1 = self.act(out1)

        if self.norm is not None:
            out = self.bn(self.conv2(out1))
        else:
            out = self.conv2(out1)

        if self.norm is not None:
            out = self.bn(self.conv3(out))
        else:
            out = self.conv3(out)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv4(out))
        else:
            out = self.conv4(out)


        out = torch.add(out, out1)
        return out

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu', norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)        

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0
        
class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

# class DownBlock(torch.nn.Module):
#     def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
#         super(DownBlock, self).__init__()
#         self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
#         self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
#         self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

#     def forward(self, x):
#       l0 = self.down_conv1(x)
#       h0 = self.down_conv2(l0)
#       l1 = self.down_conv3(h0 - x)
#       return l1 + l0

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, out_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.conv0 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(out_filter, out_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(out_filter, out_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(out_filter, out_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
      x = self.conv0(x)
      l0 = self.down_conv1(x)
      h0 = self.down_conv2(l0)
      l1 = self.down_conv3(h0 - x)
      return l1 + l0


class DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='prelu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class D_DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
            #modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)
        
        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out
             

class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out


class UpBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock_x8, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv4 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.up_conv5 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv6 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv7 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        h0 = self.up_conv1(x)
        h1 = self.up_conv2(h0)
        h2 = self.up_conv3(h1)
        l0 = self.up_conv4(h2)
        h3 = self.up_conv5(l0 - x)
        h4 = self.up_conv6(h3)
        h5 = self.up_conv7(h4)
        return h2 + h5

class DownBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock_x8, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv4 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv5 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h1 = self.down_conv2(l0)
        h2 = self.down_conv3(h1)
        h3 = self.down_conv4(h2)
        l1 = self.down_conv5(h3 - x)
        return l0 + l1

class D_DownBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock_x8, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv4 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv5 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)

    def forward(self, x):
        x1 = self.conv(x)
        l0 = self.down_conv1(x1)
        h0 = self.down_conv2(l0)
        h1 = self.down_conv3(h0)
        h2 = self.down_conv4(h1)
        l1 = self.down_conv5(h2 - x1)
        return l1 + l0


class D_UpBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock_x8, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv4 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.up_conv5 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) 
        self.up_conv6 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv7 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)       

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        h1 = self.up_conv2(h0)
        h2 = self.up_conv3(h1)
        l0 = self.up_conv4(h2)
        h3 = self.up_conv5(l0 - x)
        h4 = self.up_conv6(h3)
        h5 = self.up_conv7(h4)
        return h2 + h5

class PriorBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)


        return h_prior1

class PriorBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock2, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)


        return h_prior1

class PriorBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock3, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(4*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(5*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(6*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)


        return h_prior1
class PriorBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock4, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(5*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(6*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(7*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)


        return h_prior1
class PriorBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock5, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(6*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(7*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(8*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)


        return h_prior1
class PriorBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock5, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(6*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(7*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(8*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)
        return h_prior1

class PriorBlock6(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(PriorBlock6, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(7*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(8*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(9*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(4*64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4),1)

        h_prior1 = self.direct_up1(concat_p1)
        #out = self.output_conv1(h_prior1)
        return h_prior1
class LA_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(LA_attentionBlock, self).__init__()

        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)

        self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x):

        p1 = self.avgpool_1(x)
        #print(p1.shape, '111111111111')
        l0 = self.up_1(p1)
        #print(l0.shape, 'l0l0l0l0l0l0l0l0l0l0l0l0')
        act1 = self.act_1(x - l0)
        #print(act1.shape, 'acacacacacaacacaacac')
        out_la = x + 0.2*(act1*x)
        #print(out_la.shape, 'oooooooooooooooooooooooooooo')

        return out_la

class LA_attentionBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(LA_attentionBlock2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)

        self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x):

        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        #print(p1.shape, '111111111111')
        l0 = self.up_1(p1)
        #print(l0.shape, 'l0l0l0l0l0l0l0l0l0l0l0l0')
        act1 = self.act_1(x - l0)
        #print(act1.shape, 'acacacacacaacacaacac')
        out_la = x + 0.2*(act1*x)
        #print(out_la.shape, 'oooooooooooooooooooooooooooo')

        return out_la
class GA_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(GA_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x): 

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x*act2

        return y


class GA_res_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(GA_res_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x): 

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x*act2

        return y
        
class UpBlockP(torch.nn.Module):
    def __init__(self, num, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlockP, self).__init__()
        self.conv1 = ConvBlock(num*num_filter, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        x = self.conv1(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0



#####2019.07.22


#############2020.01.09
class MultiViewColorBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(MultiViewColorBlock, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        #self.direct_up1 = DeconvBlock(64, 64, 6, 2, 2, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        #print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(concat_p1)
        #print(x_prior1_2.shape, 'cccccccccccccccccccccccccccccc')
        
       # h_prior1 = self.direct_up1(x_prior1_2)
        #out = self.output_conv1(h_prior1)


        return x_prior1_2
class MultiViewColorBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(MultiViewColorBlock2, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = ConvBlock(5*64, 64, 3, 1, 1, activation, norm=None)
        #self.direct_up1 = DeconvBlock(64, 64, 6, 2, 2, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        #concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(x)
        #concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(x)
        #concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(x)

        concat_p1 = torch.cat((x, x_prior1, x_prior2, x_prior3, x_prior4),1)
        #print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(concat_p1)
        #print(x_prior1_2.shape, 'cccccccccccccccccccccccccccccc')
        
       # h_prior1 = self.direct_up1(x_prior1_2)
        #out = self.output_conv1(h_prior1)


        return x_prior1_2
###########
class MultiViewBlock1(torch.nn.Module):
    def __init__(self, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock1, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, 6, 2, 2, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        #print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(concat_p1)
        #print(x_prior1_2.shape, 'cccccccccccccccccccccccccccccc')
        
        h_prior1 = self.direct_up1(x_prior1_2)
        #out = self.output_conv1(h_prior1)


        return h_prior1

class MultiViewBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock2, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)
        #out = self.output_conv1(h_prior1)


        return h_prior1

class MultiViewBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock3, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(4*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(5*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(6*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(7*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)


        return h_prior1
class MultiViewBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock4, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(5*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(6*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(7*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(8*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)


        return h_prior1

class MultiViewBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock5, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(6*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(7*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(8*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(9*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)


        return h_prior1
        
class FeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class FeedbackBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0

class channel_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(channel_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x): 

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x*act2

        return y
class Color_activate_map(torch.nn.Module):
    def __init__(self, num_filter):
        super(Color_activate_map, self).__init__()
        self.conv1 = ConvBlock(num_filter, 5, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(5, 5, 3, 1, 1, activation='prelu', norm=None)
        self.conv3 = ConvBlock(5, 5, 1, 1, 0, activation='prelu', norm=None)

        self.act1 = torch.nn.Softmax()      

    def forward(self, x): 

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        act = self.act1(x3)
        y =  x1*act

        return y
        
class PDPANETFeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETFeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class PDPANETFeedbackBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETFeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.conv1(x1)
        h0 = self.up_conv1(x2)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x2)
        return h1 + h0


class PDPANETAttentionBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETAttentionBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)

        return h0


class PDPANETAttentionBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETAttentionBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)

        return h0


class RDBBlock(torch.nn.Module):
    def __init__(self, in_filter, num_filter, bias=True, activation='prelu', norm=None):
        super(RDBBlock, self).__init__()
        self.c1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c2 = ConvBlock(2*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c3 = ConvBlock(3*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c4 = ConvBlock(4*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c5 = ConvBlock(5*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c6 = ConvBlock(6*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c7 = ConvBlock(7*in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
    def forward(self, x):
        
        x1 = self.c1(x)
        cat1 = torch.cat((x, x1), 1)

        x2 = self.c2(cat1)
        cat2 = torch.cat((cat1, x2), 1)

        x3 = self.c3(cat2)
        cat3 = torch.cat((cat2, x3), 1)

        x4 = self.c4(cat3)
        cat4 = torch.cat((cat3, x4), 1)
        
        x5 = self.c5(cat4)
        cat5 = torch.cat((cat4, x5), 1)

        x6 = self.c6(cat5)
        cat6 = torch.cat((cat5, x6), 1)

        x7 = self.c7(cat6)

        out = x + x7

        return out
class RDBBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, bias=True, activation='prelu', norm=None):
        super(RDBBlock2, self).__init__()
        self.c1 = ConvBlock(2*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c2 = ConvBlock(3*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c3 = ConvBlock(4*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c4 = ConvBlock(5*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c5 = ConvBlock(6*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c6 = ConvBlock(7*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c7 = ConvBlock(8*in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
    def forward(self, x):
        
        x1 = self.c1(x)
        cat1 = torch.cat((x, x1), 1)

        x2 = self.c2(cat1)
        cat2 = torch.cat((cat1, x2), 1)

        x3 = self.c3(cat2)
        cat3 = torch.cat((cat2, x3), 1)

        x4 = self.c4(cat3)
        cat4 = torch.cat((cat3, x4), 1)
        
        x5 = self.c5(cat4)
        cat5 = torch.cat((cat4, x5), 1)

        x6 = self.c6(cat5)
        cat6 = torch.cat((cat5, x6), 1)

        x7 = self.c7(cat6)

        out = x1 + x7

        return out
class RDBBlock_s(torch.nn.Module):
    def __init__(self, in_filter, num_filter, bias=True, activation='prelu', norm=None):
        super(RDBBlock_s, self).__init__()
        self.c1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c2 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c3 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c4 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)

    def forward(self, x):
        
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        out = x + x4
        return out
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(EncoderBlock, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool1 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv1_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')



        self.conv2_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool2 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv2_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool3 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv3_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_5 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')


        self.down1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation='prelu', norm='batch')
        self.down2 = ConvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm='batch')
        self.down3 = ConvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')


        self.conv4_1 = ConvBlock(3*num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)  #256
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        s1 = torch.cat((c1_1, c1_3),1)
        c1_4 = self.conv1_4(s1)
        p1 = self.pool1(c1_4)# 128

        c2_1 = self.conv2_1(p1) #128
        c2_2 = self.conv2_2(c2_1)
        c2_3 = self.conv2_3(c2_2)
        s2 = torch.cat((p1, c2_3),1)
        c2_4 = self.conv2_4(s2)
        p2 = self.pool2(c2_4)#64

        c3_1 = self.conv3_1(p2)#64
        c3_2 = self.conv3_2(c3_1)
        c3_3 = self.conv3_3(c3_2)
        s3 = torch.cat((p2, c3_3), 1)
        c3_4 = self.conv3_4(s3)
        p3 = self.pool3(c3_4)#32
        c3_5 = self.conv3_5(p3)#32

        e1 = self.down1(c1_4)
        e2 = self.down2(c2_4)

        s4 = torch.cat((c3_5, e1, e2), 1)

        out1 = self.conv4_1(s4)
        out = self.conv4_2(out1)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(DecoderBlock, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up1 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv2_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up2 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv3_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up3 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv4_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        hr1 = self.up1(c1_3)



        c2_1 = self.conv2_1(hr1)
        c2_2 = self.conv2_2(c2_1)
        hr2 = self.up2(c2_2)

        c3_1 = self.conv3_1(hr2)
        c3_2 = self.conv3_2(c3_1)
        hr3 = self.up3(c3_2)


        out1 = self.conv4_1(hr3)
        out = self.conv4_2(out1) 
        return out

class EncoderBlock2(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(EncoderBlock2, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool1 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv1_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')



        self.conv2_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool2 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv2_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool3 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv3_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv4_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool4 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv4_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv4_5 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.down1 = ConvBlock(num_filter, num_filter, 20, 16, 2, activation='prelu', norm='batch')
        self.down2 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation='prelu', norm='batch')
        self.down3 = ConvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm='batch')
        self.down4 = ConvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')


        self.conv5_1 = ConvBlock(4*num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv5_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)  #256
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        s1 = torch.cat((c1_1, c1_3),1)
        c1_4 = self.conv1_4(s1)
        p1 = self.pool1(c1_4)# 128

        c2_1 = self.conv2_1(p1) #128
        c2_2 = self.conv2_2(c2_1)
        c2_3 = self.conv2_3(c2_2)
        s2 = torch.cat((p1, c2_3),1)
        c2_4 = self.conv2_4(s2)
        p2 = self.pool2(c2_4)#64

        c3_1 = self.conv3_1(p2)#64
        c3_2 = self.conv3_2(c3_1)
        c3_3 = self.conv3_3(c3_2)
        s3 = torch.cat((p2, c3_3), 1)
        c3_4 = self.conv3_4(s3)
        p3 = self.pool3(c3_4)#32

        c4_1 = self.conv4_1(p3)
        c4_2 = self.conv4_2(c4_1)
        c4_3 = self.conv4_3(c4_2)
        s4 = torch.cat((p3, c4_3), 1)
        c4_4 = self.conv4_4(s4)
        p4 = self.pool4(c4_4)
        c4_5 = self.conv4_5(p4)#16

        e1 = self.down1(c1_4)
        e2 = self.down2(c2_4)
        e3 = self.down3(c3_4)
        e3 = self.down4(c4_4)

        s4 = torch.cat((c4_5, e1, e2, e3), 1)

        out1 = self.conv5_1(s4)
        out = self.conv5_2(out1)
        return out


class DecoderBlock2(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(DecoderBlock2, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up1 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv2_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up2 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv3_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up3 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv4_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up4 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None) 

        self.conv5_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv5_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        hr1 = self.up1(c1_3)



        c2_1 = self.conv2_1(hr1)
        c2_2 = self.conv2_2(c2_1)
        hr2 = self.up2(c2_2)

        c3_1 = self.conv3_1(hr2)
        c3_2 = self.conv3_2(c3_1)
        hr3 = self.up3(c3_2)

        c4_1 = self.conv4_1(hr3)
        c4_2 = self.conv4_2(c4_1)
        hr4 = self.up4(c4_2)

        out1 = self.conv5_1(hr4)
        out = self.conv5_2(out1) 
        return out

class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(GeneratorBlock, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool1 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv1_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')



        self.conv2_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool2 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv2_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool3 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv3_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_5 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')


        self.deconv1 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')
        self.deconv2 = DeconvBlock(2*num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')
        self.deconv3 = DeconvBlock(2*num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')
        self.deconv4 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')

        self.conv4_1 = ConvBlock(2*num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)  #256
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        s1 = torch.cat((c1_1, c1_3),1)
        c1_4 = self.conv1_4(s1)
        p1 = self.pool1(c1_4)# 128

        c2_1 = self.conv2_1(p1) #128
        c2_2 = self.conv2_2(c2_1)
        c2_3 = self.conv2_3(c2_2)
        s2 = torch.cat((p1, c2_3),1)
        c2_4 = self.conv2_4(s2)
        p2 = self.pool2(c2_4)#64

        c3_1 = self.conv3_1(p2)#64
        c3_2 = self.conv3_2(c3_1)
        c3_3 = self.conv3_3(c3_2)
        s3 = torch.cat((p2, c3_3), 1)
        c3_4 = self.conv3_4(s3)
        p3 = self.pool3(c3_4)#32
        c3_5 = self.conv3_5(p3)#32

        dc1 = self.deconv1(c3_5)#64
        u1 = torch.cat((dc1, c3_4), 1)#64
        dc2 = self.deconv2(u1)#128
        u2 = torch.cat((dc2, c2_4), 1)#128
        dc3 = self.deconv3(u2)#256
        u3 = torch.cat((dc3, c1_4), 1)#256

        out1 = self.conv4_1(u3)#256
        out = self.conv4_2(out1)#256

        return out

class Generator_slice(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(Generator_slice, self).__init__()

        self.conv1_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.pool1 = torch.nn.AvgPool2d(2, 2, 0)
        #self.conv1_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv2_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.pool2 = torch.nn.AvgPool2d(2, 2, 0)
        #self.conv2_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.pool3 = torch.nn.AvgPool2d(2, 2, 0)
        # self.conv3_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_5 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)


        self.deconv1 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm=None)
        self.deconv2 = DeconvBlock(2*num_filter, num_filter, 6, 2, 2, activation='prelu', norm=None)
        self.deconv3 = DeconvBlock(2*num_filter, num_filter, 6, 2, 2, activation='prelu', norm=None)
        #elf.deconv4 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')

        self.conv4_1 = ConvBlock(2*num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation=None, norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)  #256
        p1 = self.pool1(c1_1)# 128

        c2_1 = self.conv2_1(p1) #128
        p2 = self.pool2(c2_1)#64

        c3_1 = self.conv3_1(p2)#64
        p3 = self.pool3(c3_1)#32
        c3_5 = self.conv3_5(p3)#32

        dc1 = self.deconv1(c3_5)#64
        u1 = torch.cat((dc1, c3_1), 1)#64
        dc2 = self.deconv2(u1)#128
        u2 = torch.cat((dc2, c2_1), 1)#128
        dc3 = self.deconv3(u2)#256
        u3 = torch.cat((dc3, c1_1), 1)#256

        out1 = self.conv4_1(u3)#256
        out = self.conv4_2(out1)#256

        return out


class MFBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MFBlock, self).__init__()

        self.conv1 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        #self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(64, 32, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3*32, 32, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4*32, 32, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv5 = DilaConvBlock(5*32, 32, 3, 1, 5, dilation=5, activation='prelu', norm=None)
        self.out_conv = ConvBlock(6*32, 64, 3, 1, 1, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x, c):

        x = torch.cat((x, c),1)

        x_prior1 = self.conv1(x)
        #concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(x_prior1)

        concat2 = torch.cat((x_prior1, x_prior2),1)

        x_prior3 = self.dilaconv3(concat2)

        concat3 = torch.cat((concat2, x_prior3),1)

        x_prior4 = self.dilaconv4(concat3)

        concat4 = torch.cat((concat3, x_prior4),1)
 
        x_prior5 = self.dilaconv5(concat4)


        concat_all = torch.cat((concat4, x_prior5), 1)

        mf = self.out_conv(concat_all)


        return mf

class MultiFusionBlock(torch.nn.Module):
    def __init__(self, num_filter, out_filter):
        super(MultiFusionBlock, self).__init__()

        self.conv1 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(2*num_filter, num_filter, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv5 = DilaConvBlock(2*num_filter, num_filter, 3, 1, 5, dilation=5, activation='prelu', norm=None)
        self.dilaconv7 = DilaConvBlock(2*num_filter, num_filter, 3, 1, 7, dilation=7, activation='prelu', norm=None)
        #self.pool1 = torch.nn.MaxPool2d(2, 2, 0)
        self.out_conv = ConvBlock(4*num_filter, out_filter, 3, 1, 1, activation='prelu', norm=None)
    

    def forward(self, x, c):

        x = torch.cat((x, c),1)

        x_1 = self.conv1(x)
        x_3 = self.dilaconv3(x)
        x_5 = self.dilaconv5(x)
        x_7 = self.dilaconv7(x)

        concat_all = torch.cat((x_1, x_3, x_5, x_7), 1)
        mf = self.out_conv(concat_all)


        return mf



class MiddleOutputBlock(torch.nn.Module):
    def __init__(self, num_filter, out_filter):
        super(MiddleOutputBlock, self).__init__()

        self.conv1 = ConvBlock(num_filter, 32, 1, 1, 0, activation='prelu', norm=None)
        self.out_conv = ConvBlock(32, out_filter, 3, 1, 1, activation='prelu', norm=None)
    
    def forward(self, x):

        x_1 = self.conv1(x)
        mo = self.out_conv(x_1)

        return mo






class SimilarityBlock(torch.nn.Module):
    def __init__(self, num_filter, out_filter, pool_filter_size, pool_stride, pool_padding):
        super(SimilarityBlock, self).__init__()

        self.conv1 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm=None)
        self.maxpool = torch.nn.MaxPool2d(pool_filter_size, pool_stride, pool_padding)# 8 4 2
        self.act = torch.nn.Sigmoid()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)

        batchSize = x.shape[0]
        channals = x.shape[1]
        w = x.shape[2]
        h = x.shape[3]

        d_x = torch.reshape(x, (batchSize, w*h, channals))
        d_x_t = torch.transpose(d_x, 1, 2)
        s_x = torch.bmm(self.act(d_x), self.act(d_x_t))
        # print(s_x.shape, '11111111111111111111111111')

        return s_x

class SimilarityBlock0(torch.nn.Module):
    def __init__(self):
        super(SimilarityBlock0, self).__init__()

    
    def forward(self, x):
        
        x= torch.nn.functional.interpolate(x, size=None, scale_factor=0.128, mode='bicubic', align_corners=None)

        N,C,H,W = x.shape
        f_1 = x.permute(0,2,3,1).view((N,H*W,C))
        s1 = torch.sqrt((f_1*f_1).sum(2).unsqueeze(2))
        f_1 = f_1/s1
        f_2 = x.view(N,C,H*W)
        s2 = torch.sqrt((f_2*f_2).sum(1).unsqueeze(1))
        f_2 = f_2/s2
        score = torch.matmul(f_1, f_2)
        score = torch.nn.functional.softmax(score, dim=2)

        return score


