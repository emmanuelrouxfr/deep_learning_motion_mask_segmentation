""" Full assembly of the parts to form the complete network """

from .unet_parts import Down, Up, OutConv, DoubleConv
from torch import nn
#from pytorch_model_summary import summary


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels=[16,32,64,128], bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernels = kernels
        self.bilinear = bilinear

        #16,32,64,128
        self.inc = DoubleConv(n_channels, kernels[0])
        self.down1 = Down(kernels[0], kernels[1])
        self.down2 = Down(kernels[1], kernels[2])
        self.down3 = Down(kernels[2], kernels[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(kernels[3], kernels[3]*2 // factor)
        self.up1 = Up(kernels[3]*2, kernels[3] // factor, bilinear)
        self.up2 = Up(kernels[3], kernels[2] // factor, bilinear)
        self.up3 = Up(kernels[2], kernels[1] // factor, bilinear)
        self.up4 = Up(kernels[1], kernels[0], bilinear)
        self.outc = OutConv(kernels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# # show input shape
# print(summary(UNet(1,2), torch.zeros((1, 1, 28, 28)), show_input=True))

# # show output shape
# print(summary(UNet(1,2), torch.zeros((1, 1, 28, 28)), show_input=False))

# # show output shape and hierarchical view of net
# print(summary(UNet(1,2), torch.zeros((1, 1, 28, 28)), show_input=False, show_hierarchical=True))
