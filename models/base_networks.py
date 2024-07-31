import torch
import math
from mmcv.ops import DeformConv2dPack as DCN
import torch.nn.functional as F



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


class Fusion_Cross_Atten_Bilinear_block(torch.nn.Module):
    def __init__(self, input_dim):
        super(Fusion_Cross_Atten_Bilinear_block, self).__init__()
        # self.conv_f = ConvBlock(2*input_dim, input_dim, 1, 1, 0, activation='prelu', norm=None)
        # self.act = torch.nn.Softmax(dim=1)
        self.down = torch.nn.MaxPool2d(8,8)
        # self.conv_up_d = DeconvBlock(input_dim, input_dim, 12, 8, 2, activation='prelu', norm=None)
        # self.conv_up_c = DeconvBlock(input_dim, input_dim, 12, 8, 2, activation='prelu', norm=None)

        self.d_Q = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.d_K = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.d_V = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)

        self.c_Q = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.c_K = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)
        self.c_V = ConvBlock(input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)

        self.act = torch.nn.Softmax(dim=1)

        self.fuse = ConvBlock(2*input_dim, input_dim, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, depth, color):

        b0, c0, h0, w0 = depth.shape
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
        # att_c = self.conv_up_c(att_c)
        att_c = F.interpolate(att_c, size=(int(h0), int(w0)), mode='bilinear', align_corners=True)

        mut_c_Q_K = self.act(torch.matmul(v_d_K, v_c_Q))
        att_d = torch.matmul(mut_c_Q_K, v_d_V).permute(0, 2, 1).contiguous().view(b, c, h, w)
        # att_d = self.conv_up_d(att_d)
        att_d = F.interpolate(att_d, size=(int(h0), int(w0)), mode='bilinear', align_corners=True)
      
        fuse_d = depth+1.5*att_d
        fuse_c = color+1.5*att_c

        fuse_d_c = torch.cat((fuse_d, fuse_c), 1)
        fuse = self.fuse(fuse_d_c)
  
        return fuse


class FeatureAlign_CA_c2f_Block5(torch.nn.Module): 
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_c2f_Block5, self).__init__()
        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv2 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv3 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv4 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen3 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen4 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_3 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_4 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        
        self.fusion0 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion1 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion2 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion3 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion4 = Fusion_Cross_Atten_Bilinear_block(out_nc)

        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[4].size(2)), int(rgb[4].size(3))

        "coarse to fine"

        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]*2], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h*0.125), int(w*0.125)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0]) ) 
        
        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]*2, offset0_up], dim=1))
        offset1_up = F.interpolate(offset1, size=(int(h*0.25), int(w*0.25)), mode='bilinear', align_corners=True)
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]) )
        
        offset2 = self.offsetGen2(torch.cat([rgb[2], depth[2]*2, offset1_up], dim=1))
        offset2_up = F.interpolate(offset2, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)
        feat_D2 = self.relu(self.dcpack_2([depth[2], offset2]) )
    
        offset3 = self.offsetGen3(torch.cat([rgb[3], depth[3]*2, offset2_up], dim=1))
        offset3_up = F.interpolate(offset3, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D3 = self.relu(self.dcpack_2([depth[3], offset3]) )
        
        offset4 = self.offsetGen3(torch.cat([rgb[4], depth[4]*2, offset3_up], dim=1))
        feat_D4 = self.relu(self.dcpack_2([depth[4], offset4]) )
        
        feat_fuse0 = self.fusion0(rgb[0], feat_D0)    
        feat_fuse1 = self.fusion1(rgb[1], feat_D1)    
        feat_fuse2 = self.fusion2(rgb[2], feat_D2)    
        feat_fuse3 = self.fusion3(rgb[3], feat_D3)   
        feat_fuse4 = self.fusion3(rgb[4], feat_D4)

        return feat_fuse0, feat_fuse1, feat_fuse2, feat_fuse3, feat_fuse4

class FeatureAlign_CA_c2f_Block4(torch.nn.Module):  
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_c2f_Block4, self).__init__()
        self.lateral_conv0 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv1 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv2 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.lateral_conv3 = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen1 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen2 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.offsetGen3 = torch.nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0)
        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_1 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_2 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
        self.dcpack_3 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)
    
        
        self.fusion0 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion1 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion2 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion3 = Fusion_Cross_Atten_Bilinear_block(out_nc)

        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[3].size(2)), int(rgb[3].size(3))

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]*2], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h*0.25), int(w*0.25)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0]) ) 
        
        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]*2, offset0_up], dim=1))
        offset1_up = F.interpolate(offset1, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]) )
        
        offset2 = self.offsetGen2(torch.cat([rgb[2], depth[2]*2, offset1_up], dim=1))
        offset2_up = F.interpolate(offset2, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D2 = self.relu(self.dcpack_2([depth[2], offset2]) )
    
        offset3 = self.offsetGen3(torch.cat([rgb[3], depth[3]*2, offset2_up], dim=1))
        feat_D3 = self.relu(self.dcpack_2([depth[3], offset3]) )
        
        
        feat_fuse0 = self.fusion0(rgb[0], feat_D0)    
        feat_fuse1 = self.fusion1(rgb[1], feat_D1)    
        feat_fuse2 = self.fusion2(rgb[2], feat_D2)    
        feat_fuse3 = self.fusion3(rgb[3], feat_D3)   

        return feat_fuse0, feat_fuse1, feat_fuse2, feat_fuse3

class FeatureAlign_CA_c2f_Block3(torch.nn.Module): 
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
        
        self.fusion0 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion1 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion2 = Fusion_Cross_Atten_Bilinear_block(out_nc)
       
        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):

        h, w = int(rgb[2].size(2)), int(rgb[2].size(3))

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]*2], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h*0.5), int(w*0.5)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0]) ) 
    
        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]*2, offset0_up], dim=1))
        offset1_up = F.interpolate(offset1, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]) )
       
        offset2 = self.offsetGen2(torch.cat([rgb[2], depth[2]*2, offset1_up], dim=1))
        feat_D2 = self.relu(self.dcpack_2([depth[2], offset2]) )
       
        feat_fuse0 = self.fusion0(rgb[0], feat_D0)    
        feat_fuse1 = self.fusion1(rgb[1], feat_D1)    
        feat_fuse2 = self.fusion2(rgb[2], feat_D2)    
        
        return feat_fuse0, feat_fuse1, feat_fuse2

class FeatureAlign_CA_c2f_Block2(torch.nn.Module): 
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
    
        self.fusion0 = Fusion_Cross_Atten_Bilinear_block(out_nc)
        self.fusion1 = Fusion_Cross_Atten_Bilinear_block(out_nc)
      
       
        self.relu = torch.nn.ReLU(inplace=True)
       

    def forward(self, rgb, depth):
    
        h, w = int(rgb[1].size(2)), int(rgb[1].size(3))

        "coarse to fine"
        offset0 = self.offsetGen0(torch.cat([rgb[0], depth[0]*2], dim=1))  # concat for offset
        offset0_up = F.interpolate(offset0, size=(int(h), int(w)), mode='bilinear', align_corners=True)
        feat_D0 = self.relu(self.dcpack_0([depth[0], offset0]) ) 

        offset1 = self.offsetGen1(torch.cat([rgb[1], depth[1]*2, offset0_up], dim=1))
        feat_D1 = self.relu(self.dcpack_1([depth[1], offset1]) )
    

        feat_fuse0 = self.fusion0(rgb[0], feat_D0)    
        feat_fuse1 = self.fusion1(rgb[1], feat_D1)    
  
        return feat_fuse0, feat_fuse1

class FeatureAlign_CA_c2f_Block1(torch.nn.Module):  
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_CA_c2f_Block1, self).__init__()

        self.offsetGen0 = torch.nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)

        self.dcpack_0 = DCN(out_nc, out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=8)


        self.fusion0 = Fusion_Cross_Atten_Bilinear_block(out_nc)

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







