import os
import torch.nn as nn
import torch.optim as optim
from models.base_networks import *

from torchvision.transforms import *
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        #### 
        elif scale_factor == 16:
          kernel = 20
          stride = 16
          padding = 2
###########depth

        self.featd01=ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featd02=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featd1=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.up1 = DeconvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)

        self.featd2=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.up2 = DeconvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.featd3=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
       

###########x1  
        self.featc1_01 = ConvBlock(3*num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featc1_02 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featc1_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # # self.featcd1_1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd1_1 = FeatureAlign_V2(base_filter, base_filter)

        self.featc1_2 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_2 = torch.nn.MaxPool2d(2, 2)
        # # self.featcd1_2 = FeatureAlign_V2(base_filter, base_filter)
        # self.featcd1_2 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.featc1_3 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_31 = torch.nn.MaxPool2d(2, 2)
        self.down1_32 = torch.nn.MaxPool2d(4, 4)
        self.featcc1_3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd1_3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd1_3 = FeatureAlign_V2(base_filter, base_filter)
        self.featc1_4 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_41 = torch.nn.MaxPool2d(2, 2)
        self.down1_42 = torch.nn.MaxPool2d(4, 4)       
        self.featcc1_4 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
 
############x2
        self.featc2_0 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd2_1=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd2_1 = FeatureAlign_V2(base_filter, base_filter)
        self.featc2_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down2_1 = torch.nn.MaxPool2d(2, 2)
        self.featcc2_1=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd2_2=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd2_2 = FeatureAlign_V2(base_filter, base_filter)
        self.featc2_2 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down2_2 = torch.nn.MaxPool2d(2, 2)
        self.featcc2_2=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
  
     
############x3
        self.featc3_0 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd3_1=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd3_1 = FeatureAlign_V2(base_filter, base_filter)
        self.featc3_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featcc3_1=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)

### Alignment by stage  
        self.fuse3 = FeatureAlign_CA_c2f_Block3(base_filter, base_filter)
        self.fuse2 = FeatureAlign_CA_c2f_Block2(base_filter, base_filter)
        self.fuse1 = FeatureAlign_CA_c2f_Block1(base_filter, base_filter)

############d-x1

        self.output_SR_conv1 = ConvBlock(3*base_filter, base_filter, 3, 1, 1, activation=None, norm=None)
        self.output_SR_conv2 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
############grad map       
        # self.generat_grad = Gradient_Map()

 
            
    def forward(self, depth, rgb):


        h0, w0 = int(rgb.size(2)), int(rgb.size(3))
#############grad map
        # grad_gt = self.generat_grad(gt)
        # grad_gtx2 = F.interpolate(depth, size=(int(0.5*h0), int(0.5*w0)), mode='bicubic', align_corners=True)
        # grad_gtx4 = F.interpolate(depth, size=(int(0.25*h0), int(0.25*w0)), mode='bicubic', align_corners=True)     
        # grad_gtall = [10*grad_gt, 10*grad_gtx2, 10*grad_gtx4]        
#############depth
      
        depth0 = F.interpolate(depth, size=(int(h0), int(w0)), mode='bicubic', align_corners=False)

        d1 = self.featd01(depth)
        d2 = self.featd02(d1)
        d3 = self.featd1(d2)
        sr1 = self.up1(d3)


        d3_up1 = F.interpolate(d3, size=(int(h0), int(w0)), mode='bilinear', align_corners=True)
        d3_up2 = F.interpolate(d3, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=True)
        d3_up3 = F.interpolate(d3, size=(int(0.25*h0), int(0.25*w0)), mode='bilinear', align_corners=True)


        d4 = self.featd2(sr1)
        sr2 = self.up2(d4)

        d4_up1 = F.interpolate(d4, size=(int(h0), int(w0)), mode='bilinear', align_corners=True)
        d4_up2 = F.interpolate(d4, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=True)

        d5 = self.featd3(sr2)



##########step0(stage by stage)

        c01 = self.featc1_01(rgb)
        c02 = self.featc1_02(c01)

 
##########step1
        ##

        c11_x4 = self.featc1_1(c02)
        # c12_x4 = self.featcd1_1(c11_x4, d5)[0]
        # e1= self.featcd1_1(c11_x4, d5)[1]
       
        # c12_x4 = c11_x4+d5
        # c12_x4 = self.featcd1_1(c12_x4)

        c12_x4 = self.fuse1 (c11_x4, d5)

##########step2
        c21_x4 = self.featc1_2(c12_x4)
        c21down1_x4 =self.down1_2(c21_x4)
        # c22_x4 = self.featcd1_2(c21_x4, d4_up1)[0]


        c21_x2 = self.featc2_0(c21down1_x4)
        # c22_x2 = self.featcd2_1(c21_x2,d4_up2)[0]
       
        
       
        # c22_x2  = c21_x2+d4_up2
        # c22_x4  = c21_x4+d4_up1
        # c22_x4 = self.featcd1_2(c22_x4)
        # c22_x2 = self.featcd2_1(c22_x2)

        rgb2 = [c21_x2, c21_x4]
        depth2 = [ d4_up2, d4_up1]
        c22_x2 , c22_x4 = self.fuse2(rgb2, depth2)
        
##########step3
        c31_x4 = self.featc1_3(c22_x4)
        c31down1_x4 = self.down1_31(c31_x4)
        c31down2_x4 = self.down1_32(c31_x4)  

        c31_x2 = self.featc2_1(c22_x2)
        c31down_x2 = self.down2_1(c31_x2) 
        c31up_x2 = F.interpolate(c31_x2, size=(h0, w0), mode='bilinear', align_corners=True)

        c32_x4 = self.featcc1_3(c31_x4+c31up_x2)
        # c33_x4 = self.featcd1_3(c32_x4, d3_up1)[0]
        # e4= self.featcd1_3(c32_x4, d3_up1)[1]

        c32_x2 = self.featcc2_1(c31_x2+c31down1_x4)
        # c33_x2 = self.featcd2_2(c32_x2, d3_up2)[0]
        # e5= self.featcd2_2(c32_x2, d3_up2)[1]

        c31_lr = self.featc3_0(c31down2_x4+c31down_x2)
        # c32_lr = self.featcd3_1(c31_lr, d3_up3)[0]
        # e6 = self.featcd3_1(c31_lr, d3_up3)[1]

        # c32_lr = c31_lr + d3_up3
        # c33_x2 = c32_x2 + d3_up2
        # c33_x4 = c32_x4 + d3_up1

        # c33_x4 = self.featcd1_3(c33_x4)
        # c33_x2 = self.featcd2_2(c33_x2)
        # c32_lr = self.featcd3_1(c32_lr)

        rgb3 = [c31_lr, c32_x2, c32_x4]
        depth3 = [d3_up3, d3_up2, d3_up1]
        
        c32_lr,  c33_x2 , c33_x4 = self.fuse3(rgb3, depth3)
        

##########step4
        
        c41_x4 = self.featc1_4(c33_x4)
        c41_x2 = self.featc2_2(c33_x2)
        c41_lr = self.featc3_1(c32_lr)
        c41down1_x4 = self.down1_41(c41_x4)
        c41down2_x4 = self.down1_42(c41_x4)
        c41down_x2 =self.down2_2(c41_x2)
        c41up_x2 = F.interpolate(c41_x2, size=(h0, w0), mode='bilinear', align_corners=True)
        c41up1_lr = F.interpolate(c41_lr, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=True)
        c41up2_lr = F.interpolate(c41_lr, size=(h0, w0), mode='bilinear', align_corners=True)
        sr1 = self.featcc1_4(c41_x4+c41up_x2+c41up2_lr)
        sr2 = self.featcc2_2(c41_x2+c41down1_x4+c41up1_lr)
        sr3 = self.featcc3_1(c41_lr+c41down2_x4+c41down_x2)

#########reconstruction
        hr1 = sr1
        hr2 = F.interpolate(sr2, size=(h0, w0), mode='bilinear', align_corners=True)
        hr3 = F.interpolate(sr3, size=(h0, w0), mode='bilinear', align_corners=True)
       
        hr_all = torch.cat((hr1, hr2, hr3), 1)

    

       
        sr_out = self.output_SR_conv1(hr_all)
        sr_out = self.output_SR_conv2(sr_out)
        
        
        
        return sr_out+depth0
        