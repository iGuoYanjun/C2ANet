import os
import torch.nn as nn
import torch.optim as optim
from models.base_networks import *
# from models.generate_grad import *
from torchvision.transforms import *
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor):
        super(Net, self).__init__()
        
 
###########depth
        ###shallow
        self.featd01=ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featd02=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featd0=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##x2
        self.up1 = DeconvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.featd1=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##x4
        self.up2 = DeconvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.featd2=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##x8
        self.up3 = DeconvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.featd3=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
       
        ##x16
        self.up4 = DeconvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.featd4=ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        
###########x1  
        ##shallow
        self.featc1_01 = ConvBlock(3*num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featc1_02 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##block1
        self.featc1_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd1_1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd1_1 = FeatureAlign_V2(base_filter, base_filter)
        ##block2
        self.featc1_2 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_2 = torch.nn.MaxPool2d(2, 2)
        # self.featcd1_2 = FeatureAlign_V2(base_filter, base_filter)
        ##self.featcd1_2 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        ##block3
        self.featc1_3 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_31 = torch.nn.MaxPool2d(2, 2)
        self.down1_32 = torch.nn.MaxPool2d(4, 4)
        self.featcc1_3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd1_3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd1_3 = FeatureAlign_V2(base_filter, base_filter)
        ##block4
        self.featc1_4 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_41 = torch.nn.MaxPool2d(2, 2)
        self.down1_42 = torch.nn.MaxPool2d(4, 4)    
        self.down1_43 = torch.nn.MaxPool2d(8, 8)      
        self.featcc1_4 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd1_4 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.featcd1_4 = FeatureAlign_V2(base_filter, base_filter)
        ##block5
        self.featc1_5 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_51 = torch.nn.MaxPool2d(2, 2)
        self.down1_52 = torch.nn.MaxPool2d(4, 4)    
        self.down1_53 = torch.nn.MaxPool2d(8, 8)  
        self.down1_54 = torch.nn.MaxPool2d(16, 16)  
        self.featcc1_5 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ####block6
        self.featc1_6 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down1_61 = torch.nn.MaxPool2d(2, 2)
        self.down1_62 = torch.nn.MaxPool2d(4, 4)    
        self.down1_63 = torch.nn.MaxPool2d(8, 8)  
        self.down1_64 = torch.nn.MaxPool2d(16, 16)  
        self.featcc1_6 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        


############x2
        ##shallow
        self.featc2_0 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd2_0=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd2_0 = FeatureAlign_V2(base_filter, base_filter)
        ###block1
        self.featc2_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down2_1 = torch.nn.MaxPool2d(2, 2)
        self.featcc2_1=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd2_1=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd2_1 = FeatureAlign_V2(base_filter, base_filter)
        ###block2
        self.featc2_2 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down2_21 = torch.nn.MaxPool2d(2, 2)
        self.down2_22 = torch.nn.MaxPool2d(4, 4)
        self.featcc2_2=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        ##self.featcd2_2=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd2_2 = FeatureAlign_V2(base_filter, base_filter)
        ###block3
        self.featc2_3 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down2_31 = torch.nn.MaxPool2d(2, 2)
        self.down2_32 = torch.nn.MaxPool2d(4, 4)
        self.down2_33 = torch.nn.MaxPool2d(8, 8)
        self.featcc2_3=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        ###block4
        self.featc2_4 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down2_41 = torch.nn.MaxPool2d(2, 2)
        self.down2_42 = torch.nn.MaxPool2d(4, 4)
        self.down2_43 = torch.nn.MaxPool2d(8, 8)
        self.featcc2_4=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)

############x3
        ###shallow
        self.featc3_0 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd3_0=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd3_0 = FeatureAlign_V2(base_filter, base_filter)
        ###block1
        self.featc3_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down3_1 = torch.nn.MaxPool2d(2, 2)
        self.featcc3_1=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd3_0=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd3_0 = FeatureAlign_V2(base_filter, base_filter)
        ###block2
        self.featc3_2 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down3_21 = torch.nn.MaxPool2d(2, 2)
        self.down3_22 = torch.nn.MaxPool2d(4, 4)
        self.featcc3_2=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ###block3
        self.featc3_3 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down3_31 = torch.nn.MaxPool2d(2, 2)
        self.down3_32 = torch.nn.MaxPool2d(4, 4)
        self.featcc3_3=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)

   
############x4
        ###shallow
        self.featc4_0 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd4_0=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd4_0 = FeatureAlign_V2(base_filter, base_filter)
        ###block1
        self.featc4_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down4_1 = torch.nn.MaxPool2d(2, 2)
        self.featcc4_1=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ####block2
        self.featc4_2 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.down4_2 = torch.nn.MaxPool2d(2, 2)
        self.featcc4_2=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)


############x5
        ###shallow
        self.featc5_0 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        ##self.featcd4_0=ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featcd4_0 = FeatureAlign_V2(base_filter, base_filter)
        ###block1
        self.featc5_1 = ResnetBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featcc5_1=ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)



############fusion
        self.fuse5 = FeatureAlign_CA_c2f_Block5(base_filter, base_filter)
        self.fuse4 = FeatureAlign_CA_c2f_Block4(base_filter, base_filter)
        self.fuse3 = FeatureAlign_CA_c2f_Block3(base_filter, base_filter)
        self.fuse2 = FeatureAlign_CA_c2f_Block2(base_filter, base_filter)
        self.fuse1 = FeatureAlign_CA_c2f_Block1(base_filter, base_filter)

############d-x1

        self.output_SR_conv1 = ConvBlock(5*base_filter, base_filter, 3, 1, 1, activation=None, norm=None)

        self.output_SR_conv2 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)


 
            
    def forward(self, depth, rgb):
#############depth
        h0, w0 = int(rgb.size(2)), int(rgb.size(3))
        depth0 = F.interpolate(depth, size=(int(h0), int(w0)), mode='bicubic', align_corners=False)
        # depth1 = F.interpolate(depth0, size=(int(h0*0.25), int(w0*0.25)), mode='bicubic', align_corners=True)ß
        # depth2 = F.interpolate(depth0, size=(int(h0*0.5), int(w0*0.5)), mode='bicubic', align_corners=True)

        d1 = self.featd01(depth)
        d2 = self.featd02(d1)
        d3 = self.featd0(d2)
        d3_up1 = F.interpolate(d3, size=(int(h0), int(w0)), mode='bilinear', align_corners=False)
        d3_up2 = F.interpolate(d3, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=False)
        d3_up3 = F.interpolate(d3, size=(int(0.25*h0), int(0.25*w0)), mode='bilinear', align_corners=False)
        d3_up4 = F.interpolate(d3, size=(int(0.125*h0), int(0.125*w0)), mode='bilinear', align_corners=False)
        d3_up5 = F.interpolate(d3, size=(int(0.0625*h0), int(0.0625*w0)), mode='bilinear', align_corners=False)

        sr1 = self.up1(d3)
        d4 = self.featd1(sr1)
        d4_up1 = F.interpolate(d4, size=(int(h0), int(w0)), mode='bilinear', align_corners=False)
        d4_up2 = F.interpolate(d4, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=False)
        d4_up3 = F.interpolate(d4, size=(int(0.25*h0), int(0.25*w0)), mode='bilinear', align_corners=False)           
        d4_up4 = F.interpolate(d4, size=(int(0.125*h0), int(0.125*w0)), mode='bilinear', align_corners=False)

        sr2 = self.up2(d4)
        d5 = self.featd2(sr2)
        d5_up1 = F.interpolate(d5, size=(int(h0), int(w0)), mode='bilinear', align_corners=False)
        d5_up2 = F.interpolate(d5, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=False)
        d5_up3 = F.interpolate(d5, size=(int(0.25*h0), int(0.25*w0)), mode='bilinear', align_corners=False)

        sr3 = self.up3(d5)
        d6 = self.featd3(sr3)
        d6_up1 = F.interpolate(d6, size=(int(h0), int(w0)), mode='bilinear', align_corners=False)
        d6_up2 = F.interpolate(d6, size=(int(0.5*h0), int(0.5*w0)), mode='bilinear', align_corners=False)
       
        sr4 = self.up4(d6)
        d7 = self.featd3(sr4)
        d7_up1 = F.interpolate(d7, size=(int(h0), int(w0)), mode='bilinear', align_corners=False)
     

##########step0(stage by stage)

        c01 = self.featc1_01(rgb)
        c02 = self.featc1_02(c01)


##########step1
        

        c11_x16 = self.featc1_1(c02)

        # c12_x16 = self.featcd1_1(c11_x16, d7_up1)
        c12_x16 = self.fuse1(c11_x16, d7_up1)


##########step2
        c21_x16 = self.featc1_2(c12_x16)
        c21down1_x16 =self.down1_2(c21_x16)
        # c22_x16 = self.featcd1_2(c21_x16, d6_up1)

        c21_x8 = self.featc2_0(c21down1_x16)
        # c22_x8 = self.featcd2_0(c21_x8, d6_up2)

        rgb2 = [c21_x8, c21_x16]
        depth2 = [d6_up2, d6_up1] 

        
        c22_x8, c22_x16 = self.fuse2(rgb2,depth2)

##########step3
        ##x16branch
        c31_x16 = self.featc1_3(c22_x16)
        c31down1_x16 = self.down1_31(c31_x16)
        c31down2_x16 = self.down1_32(c31_x16)  
        ##x8
        c31_x8 = self.featc2_1(c22_x8)
        c31down_x8 = self.down2_1(c31_x8) 
        c31up_x8 = F.interpolate(c31_x8, size=(h0, w0), mode='bilinear', align_corners=False)
        ##x4
        c31_x4 = self.featc3_0(c31down2_x16+c31down_x8)
        # c32_x2 = self.featcd3_0(c31_x2, d4_up3)
        ##x16branch
        c32_x16 = self.featcc1_3(c31_x16+c31up_x8)
        # c33_x8 = self.featcd1_3(c32_x8, d4_up1)
        ##x8branch
        c32_x8 = self.featcc2_1(c31_x8+c31down1_x16)
        # c33_x4 = self.featcd2_1(c32_x4, d4_up2)

        rgb3 = [c31_x4,c32_x8, c32_x16]
        depth3 = [d5_up3, d5_up2, d5_up1] 

        
        c32_x4,c33_x8, c33_x16  = self.fuse3(rgb3,depth3)
        

##########step4

        ##x16branch
        c41_x16 = self.featc1_4(c33_x16)
        c41down1_x16 = self.down1_41(c41_x16)
        c41down2_x16 = self.down1_42(c41_x16)
        c41down3_x16 = self.down1_43(c41_x16)
        ##x8branch
        c41_x8 = self.featc2_2(c33_x8)
        c41down1_x8 =self.down2_21(c41_x8)
        c41down2_x8 =self.down2_22(c41_x8)
        c41up_x8 = F.interpolate(c41_x8, size=(h0, w0), mode='bilinear', align_corners=False)
        ##x4branch
        c41_x4 = self.featc3_1(c32_x4)
        c41down_x4 =self.down3_1(c41_x4)
        c41up1_x4 = F.interpolate(c41_x4, size=(h0, w0), mode='bilinear', align_corners=False)
        c41up2_x4 = F.interpolate(c41_x4, size=(int(h0*0.5),int(w0*0.5) ), mode='bilinear', align_corners=False)
        ##x2 branch
        c41_x2 =self.featc4_0(c41down3_x16+c41down2_x8+c41down_x4)
        # c42_lr =self.featcd4_0(c41_lr, d3_up4)


        ##x16branch
        c42_x16 = self.featcc1_4(c41_x16+c41up_x8+c41up1_x4)
        # c43_x8 = self.featcd1_4(c42_x8, d3_up1)
        ##x8branch
        c42_x8 = self.featcc2_2(c41_x8+c41down1_x16+c41up2_x4)
        # c43_x4 = self.featcd2_2(c42_x4, d3_up2)
        ##x4branch
        c42_x4 = self.featcc3_1(c41_x4+c41down2_x16+c41down1_x8)
        # c43_x2 = self.featcd2_2(c42_x2, d3_up3)

        rgb4 = [c41_x2, c42_x4,c42_x8, c42_x16]
        depth4 = [d4_up4, d4_up3, d4_up2, d4_up1] 

        
        c42_x2, c43_x4,c43_x8, c43_x16  = self.fuse4(rgb4,depth4)

##########step5

        ##x16branch
        c51_x16 = self.featc1_5(c43_x16)
        c51down1_x16 = self.down1_51(c51_x16)
        c51down2_x16 = self.down1_52(c51_x16)
        c51down3_x16 = self.down1_53(c51_x16)
        c51down4_x16 = self.down1_54(c51_x16)
        ##x8branch
        c51_x8 = self.featc2_3(c43_x8)
        c51down1_x8 =self.down2_31(c51_x8)
        c51down2_x8 =self.down2_32(c51_x8)
        c51down3_x8 =self.down2_33(c51_x8)
        c51up_x8 = F.interpolate(c51_x8, size=(h0, w0), mode='bilinear', align_corners=False)
        ##x4branch
        c51_x4 = self.featc3_2(c43_x4)
        c51down1_x4 =self.down3_21(c41_x4)
        c51down2_x4 =self.down3_22(c41_x4)
        c51up1_x4 = F.interpolate(c51_x4, size=(h0, w0), mode='bilinear', align_corners=False)
        c51up2_x4 = F.interpolate(c51_x4, size=(int(h0*0.5),int(w0*0.5)), mode='bilinear', align_corners=False)
        ##x2
        c51_x2 =self.featc4_1(c42_x2)
        c51up1_x2 = F.interpolate(c51_x2, size=(h0, w0), mode='bilinear', align_corners=False)
        c51up2_x2 = F.interpolate(c51_x2, size=(int(h0*0.5),int(w0*0.5)), mode='bilinear', align_corners=False)
        c51up3_x2 = F.interpolate(c51_x2, size=(int(h0*0.25),int(w0*0.25)), mode='bilinear', align_corners=False)
        c51_down_x2 =self.down4_1(c51_x2)
        ######lr
        c51_lr =self.featc5_0(c51down4_x16+c51down3_x8+c51down2_x4+c51_down_x2)
        
        ##x16branch
        c52_x16 = self.featcc1_5(c51_x16+c51up_x8+c51up1_x4+c51up1_x2)
        # c43_x8 = self.featcd1_4(c42_x8, d3_up1)
        ##x8branch
        c52_x8 = self.featcc2_3(c51_x8+c51down1_x16+c51up2_x4+c51up2_x2)
        # c43_x4 = self.featcd2_2(c42_x4, d3_up2)
        ##x4branch
        c52_x4 = self.featcc3_2(c51_x4+c51down2_x16+c51down1_x8+c51up3_x2)
        # c43_x2 = self.featcd2_2(c42_x2, d3_up3)
        ##x2branch
        c52_x2 = self.featcc4_1(c51_x2+c51down3_x16+c51down2_x8+c51down1_x4)

        rgb5 = [c51_lr, c52_x2, c52_x4,c52_x8, c52_x16]
        depth5 = [d3_up5, d3_up4, d3_up3, d3_up2, d3_up1] 

        
        c53_lr,c53_x2, c53_x4,c53_x8, c53_x16  = self.fuse5(rgb5,depth5)


##########step6

        ##x16branch
        c61_x16 = self.featc1_6(c53_x16)
        c61down1_x16 = self.down1_61(c61_x16)
        c61down2_x16 = self.down1_62(c61_x16)
        c61down3_x16 = self.down1_63(c61_x16)
        c61down4_x16 = self.down1_64(c61_x16)
        ##x8branch
        c61_x8 = self.featc2_4(c53_x8)
        c61down1_x8 =self.down2_41(c61_x8)
        c61down2_x8 =self.down2_42(c61_x8)
        c61down3_x8 =self.down2_43(c61_x8)
        c61up_x8 = F.interpolate(c61_x8, size=(h0, w0), mode='bilinear', align_corners=False)
        ##x4branch
        c61_x4 = self.featc3_3(c53_x4)
        c61down1_x4 =self.down3_31(c61_x4)
        c61down2_x4 =self.down3_32(c61_x4)
        c61up1_x4 = F.interpolate(c61_x4, size=(h0, w0), mode='bilinear', align_corners=False)
        c61up2_x4 = F.interpolate(c61_x4, size=(int(h0*0.5),int(w0*0.5)), mode='bilinear', align_corners=False)
        ##x2
        c61_x2 =self.featc4_2(c53_x2)
        c61up1_x2 = F.interpolate(c61_x2, size=(h0, w0), mode='bilinear', align_corners=False)
        c61up2_x2 = F.interpolate(c61_x2, size=(int(h0*0.5),int(w0*0.5)), mode='bilinear', align_corners=False)
        c61up3_x2 = F.interpolate(c61_x2, size=(int(h0*0.25),int(w0*0.25)), mode='bilinear', align_corners=False)
        c61_down_x2 =self.down4_2(c61_x2)
        ######lr
        c61_lr =self.featc5_1(c53_lr)
        c61up1_lr = F.interpolate(c61_lr, size=(h0, w0), mode='bilinear', align_corners=False)
        c61up2_lr = F.interpolate(c61_lr, size=(int(h0*0.5),int(w0*0.5)), mode='bilinear', align_corners=False)
        c61up3_lr = F.interpolate(c61_lr, size=(int(h0*0.25),int(w0*0.25)), mode='bilinear', align_corners=False)
        c61up4_lr = F.interpolate(c61_lr, size=(int(h0*0.125),int(w0*0.125)), mode='bilinear', align_corners=False)

        ##x16branch
        SR1 = self.featcc1_6(c61_x16+c61up_x8+c61up1_x4+c61up1_x2+c61up1_lr)
        ##x8branch
        SR2 = self.featcc2_4(c61_x8+c61down1_x16+c61up2_x4+c61up2_x2+c61up2_lr)
        ##x4branch
        SR3 = self.featcc3_3(c61_x4+c61down2_x16+c61down1_x8+c61up3_x2+c61up3_lr)
        ##x2branch
        SR4 = self.featcc4_2(c61_x2+c61down3_x16+c61down2_x8+c61down1_x4+c61up4_lr)
        ##lr branch
        SR5 = self.featcc5_1(c61_lr+c61down4_x16+c61down3_x8+c61down2_x4+c61_down_x2)

         
        # ##x8branch
        # SR1 = self.featcc1_5(c51_x8+c51up_x4+c51up1_x2+c51up1_lr)
        # ##x4branch
        # SR2 = self.featcc2_3(c51_x4+c51down1_x8+c51up2_x2+c51up2_lr)
        # ##x2branch
        # SR3 = self.featcc3_2(c51_x2+c51down2_x8+c51down1_x4+c51up3_lr)
        # ##lrbranch
        # SR4 = self.featcc4_1(c51_lr+c51down3_x8+c51down2_x4+c51down_x2)

#########reconstruction
        hr1 = SR1
        hr2 = F.interpolate(SR2, size=(h0, w0), mode='bilinear', align_corners=False)
        hr3 = F.interpolate(SR3, size=(h0, w0), mode='bilinear', align_corners=False)
        hr4 = F.interpolate(SR4, size=(h0, w0), mode='bilinear', align_corners=False)
        hr5 = F.interpolate(SR5, size=(h0, w0), mode='bilinear', align_corners=False)
        hr_all = torch.cat((hr1, hr2, hr3, hr4, hr5), 1)

        sr_out1 = self.output_SR_conv1(hr_all)
        sr_out2 = self.output_SR_conv2(sr_out1)
        return sr_out2+depth0