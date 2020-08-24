import torch
import torch.nn as nn
import torch.nn.functional as F
# =====================================================
# Multi-task Neural Networks with Spatial
# Activation for Retinal Vessel
# Segmentation and Artery/Vein
# Classification
# MICCAI 2019
# Wenao Ma, Shuang Yu, Kai Ma, Jiexiang Wang and et al
# ====================================================


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, step, identity_addition=False, no_conv_layers=2):

        super(ConvBlock, self).__init__()

        self.identity_addition = identity_addition
        self.no_conv_layer = no_conv_layers

        if kernel == 3:

            kernel_size = 3
            padding_size = 1

        elif kernel == 7:

            kernel_size = 7
            padding_size = 3

        if self.no_conv_layer == 1:

            self.main_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=step, padding=padding_size, bias=False),
                nn.BatchNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

        elif self.no_conv_layer == 2:

            self.main_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=step, padding=padding_size, bias=False),
                nn.BatchNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False),
                nn.BatchNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

        if self.identity_addition is True:

            self.identity_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=step, padding=0, bias=False)

    def forward(self, x):

        if self.identity_addition is True:

            output = self.identity_mapping(x) + self.main_block(x)

        else:

            output = self.main_block(x)

        return output


class PointBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(PointBlock, self).__init__()

        self.main_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, affine=True)
        )

    def forward(self, x):

        output = self.main_block(x)

        return output


class ActBlock(nn.Module):

    def __init__(self):

        super(ActBlock, self).__init__()

    def forward(self, x):

        output = x - torch.tensor(0.5)
        output = torch.exp(torch.tensor(-1)*output**2)
        output = output + torch.tensor(1.0) - torch.exp(torch.tensor(-0.25))

        return output


class OutputBlock(nn.Module):

    def __init__(self, output_dim):

        super(OutputBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels=64, out_channels=32, kernel=3, step=1, identity_addition=False, no_conv_layers=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=32, kernel=3, step=1, identity_addition=False, no_conv_layers=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid1 = nn.Sigmoid()

        self.activation = ActBlock()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        output = torch.cat([x1, x2], dim=1)

        x2 = self.conv3(x2)

        x2 = self.sigmoid1(x2)

        x2 = self.activation(x2)

        output = output*x2

        output = self.conv4(output)

        output = self.sigmoid2(output)

        return output


class MTSARVSnet(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(MTSARVSnet, self).__init__()
        # ================================
        # Encoder:
        # ================================
        # first two convolutional layers:
        self.conv1 = ConvBlock(in_channels=input_dim, out_channels=32, kernel=3, step=1, identity_addition=False, no_conv_layers=2)
        # 7 x 7 convolutional layer:
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel=7, step=2, identity_addition=False, no_conv_layers=1)
        # first residual block:
        self.conv3 = ConvBlock(in_channels=64, out_channels=64, kernel=3, step=2, identity_addition=True, no_conv_layers=2)
        # second residual block:
        self.conv4 = ConvBlock(in_channels=64, out_channels=128, kernel=3, step=2, identity_addition=True, no_conv_layers=2)
        # third residual block:
        self.conv5 = ConvBlock(in_channels=128, out_channels=256, kernel=3, step=1, identity_addition=True, no_conv_layers=2)
        # ================================
        # Decoder:
        # ================================
        self.dconv11 = ConvBlock(in_channels=256, out_channels=128, kernel=3, step=1, identity_addition=False, no_conv_layers=1)
        self.dconv12 = ConvBlock(in_channels=256, out_channels=128, kernel=3, step=1, identity_addition=False, no_conv_layers=2)

        self.dconv21 = ConvBlock(in_channels=128, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=1)
        self.dconv22 = ConvBlock(in_channels=128, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=2)

        self.dconv31 = ConvBlock(in_channels=64, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=1)
        self.dconv32 = ConvBlock(in_channels=128, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=2)

        self.output_main = OutputBlock(output_dim=output_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.downsample = nn.MaxPool2d(2)

        # =================
        # deep supervision
        # =================
        self.sideoutput1 = OutputBlock(output_dim=output_dim)
        self.sideoutput2 = OutputBlock(output_dim=output_dim)
        self.sideoutput3 = OutputBlock(output_dim=output_dim)

        self.deep_sup2 = ConvBlock(in_channels=64, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=1)
        self.deep_sup31 = ConvBlock(in_channels=128, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=1)
        self.deep_sup32 = ConvBlock(in_channels=64, out_channels=64, kernel=3, step=1, identity_addition=False, no_conv_layers=1)

    def forward(self, x):

        x = self.conv1(x)
        x1 = self.conv2(x)
        side_output1 = self.upsample(x1)
        side_output1 = self.sideoutput1(side_output1)

        # x2 = self.downsample(x1)
        x2 = self.conv3(x1)
        side_output2 = self.upsample(x2)
        side_output2 = self.deep_sup2(side_output2)
        side_output2 = self.upsample(side_output2)
        side_output2 = self.sideoutput2(side_output2)

        x3 = self.conv4(x2)
        side_output3 = self.upsample(x3)
        side_output3 = self.deep_sup31(side_output3)
        side_output3 = self.upsample(side_output3)
        side_output3 = self.deep_sup32(side_output3)
        side_output3 = self.upsample(side_output3)
        side_output3 = self.sideoutput3(side_output3)

        x4 = self.conv5(x3)
        output = self.upsample(x4)

        if output.size()[2] != x3.size()[2]:

            diffY = torch.tensor([x3.size()[2] - output.size()[2]])
            diffX = torch.tensor([x3.size()[3] - output.size()[3]])
            #
            output = F.pad(output, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        output = self.dconv11(output)
        output = torch.cat([output, x3], dim=1)
        output = self.dconv12(output)

        output = self.upsample(output)
        output = self.dconv21(output)
        output = torch.cat([output, x2], dim=1)
        output = self.dconv22(output)

        output = self.upsample(output)
        output = self.dconv31(output)
        output = torch.cat([output, x1], dim=1)
        output = self.dconv32(output)

        output = self.upsample(output)
        output = self.output_main(output)

        return output, side_output1, side_output2, side_output3
