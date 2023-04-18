import torch.nn as nn
import torch
import numpy as np
import functools
from torch.nn import init
import torch.nn.functional as F

#Discriminator
class NetD(nn.Module):
    def __init__(self, ndf=32, input_dim=3):
        super(NetD, self).__init__()

        self.input_pro = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=1, padding=2, bias=False)
        # Layer1 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True)

        )

        # Layer2 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer3 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer4 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer5 16
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer6 8
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer7 4
        self.layer7 = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=4, stride=4, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 5, 1, 1, 0, bias=False),
            nn.Sigmoid()

        )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self,x):
        x = self.input_pro(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        return out7.reshape(x.shape[0],-1)


