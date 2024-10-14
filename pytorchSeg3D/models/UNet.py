import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Double_Conv


# class Unet_module(nn.Module):
# 
#     def __init__(self, kernel_size, channel_list, down_up='down'):
#         super(Unet_module, self).__init__()
#         self.conv = Double_Conv(kernel_size, channel_list)
# 
#         if down_up == 'down':
#             self.sample = nn.MaxPool3d(2, 2)
#         else:
#             self.sample = nn.Sequential(nn.ConvTranspose3d(channel_list[2], channel_list[2], kernel_size,
#                                                            2, (kernel_size - 1) // 2, 1),
#                                         nn.ReLU())
# 
#     def forward(self, x):
#         x = self.conv(x)
# 
#         next_layer = self.sample(x)
# 
#         return next_layer, x

class Unet_module(nn.Module):
    
    def __init__(self, kernel_size, channel_list, down_up='down'):
        super(Unet_module, self).__init__()
        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])

        if down_up == 'down':
            self.sample = nn.MaxPool3d(2, 2)
        else:
            self.sample = nn.Sequential(nn.ConvTranspose3d(channel_list[2], channel_list[2], kernel_size,
                                                        2, (kernel_size - 1) // 2, 1),
                                        nn.ReLU())
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        next_layer = self.sample(x)

        return next_layer, x

class UNet(nn.Module):

    def __init__(self, kernel_size, in_channel=1, out_channel=2):
        super(UNet, self).__init__()

        self.encoder1 = Unet_module(kernel_size, (in_channel, 32, 64))
        self.encoder2 = Unet_module(kernel_size, (64, 64, 128))
        self.encoder3 = Unet_module(kernel_size, (128,128,256))

        self.decoder1 = Unet_module(kernel_size, (256, 256, 512), down_up='up')
        self.decoder2 = Unet_module(kernel_size, (768, 256, 256), down_up='up')
        self.decoder3 = Unet_module(kernel_size, (384, 128, 128), down_up='up')
        self.decoder4 = Unet_module(kernel_size, (192, 64, 64), down_up='up')

        self.last_conv = nn.Conv3d(64, out_channel, 1, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print('input', x.shape)
        # 下采样路径
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)

        # 上采样路径
        x, _ = self.decoder1(x)
        x = torch.cat([x, skip3], dim=1)
        x, _ = self.decoder2(x)
        x = torch.cat([x, skip2], dim=1)
        x, _ = self.decoder3(x)
        x = torch.cat([x, skip1], dim=1)
        _, x = self.decoder4(x)

        # 最终卷积层
        output = self.last_conv(x)
        # print('output', output.shape)
        output = self.softmax(output)
        return output



class sUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(sUNet, self).__init__()
        self.training = training

        # Encoder
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # Decoder
        self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 16, 3, stride=1, padding=1)

        #Connected
        self.map4 = nn.Sequential(
            nn.Conv3d(16, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear',align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear',align_corners=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        e1 = out
        # print('e1', e1.shape)
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        e2 = out
        # print('e2', e2.shape)
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        e3 = out
        # print('e3', e3.shape)
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        e4 = out
        # print('e4', e4.shape)
        # print('t4', t4.shape)
        out = F.relu(F.max_pool3d(self.encoder5(out),2,2))

        # e5 = out
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        output1 = self.map1(out)
        # print('output1',output1.shape)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear'))
        # print('d1',out.shape)
        out = torch.add(out, e3)
        # print('a1',out.shape)
        output2 = self.map2(out)
        # print('output2',output2.shape)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear'))
        # print('d2',out.shape)
        out = torch.add(out, e2)
        # print('a2',out.shape)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear'))
        # print('d3',out.shape)
        out = torch.add(out, e1)
        # print('a3',out.shape)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        # print('d4',out.shape)
        output4 = self.map4(out)
        # print('output4', output4.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

