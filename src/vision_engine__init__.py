import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        out = self.relu(g1 + x1)
        out = self.psi(out)
        return x * out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        
        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        
        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        return torch.sigmoid(self.Conv_1x1(d2))
