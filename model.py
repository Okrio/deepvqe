import torch
# import torchaudio
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding) -> None:
        super().__init__()
        self.resid = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ELU())
        
    def forward(self, x):
        x1 = self.resid(x)
        out = x1 + x
        return out


class PixelConv(nn.Module):
    def __init__(self, in_ch, up_scaler, kernel_size, stride, padding) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*up_scaler, stride=stride, kernel_size=kernel_size, padding=padding)

        # self.pixel = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, stride=stride, kernel_size=kernel_size, padding=padding),
        #                            nn.PixelShuffle(2*in_ch))
        
    def forward(self, x, upscale):
        # b c t f
        b,c,t,f = x.shape
        out1 = self.conv2d(x)
        out1 = out1[...,:-1,:]
        out2 = torch.chunk(out1, upscale, dim=1)
        out3 = torch.stack(out2, -1)
        out4 = out3.reshape(b,c,t,-1)

        # out = self.pixel(x)
        return out4


class AlignBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.pointConv = nn.Conv2d()

    def forward(self, x, y):
        pass
        return 0


class deepvqe(nn.Module):
    def __init__(self, near_in_channel=[2, 64, 128], far_in_channel=[2, 32, 128], kernel_size=[4, 3], stride=[1, 2]) -> None:
        super().__init__()
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[0], out_channels=near_in_channel[1], kernel_size=kernel_size, stride=stride),
                                     nn.BatchNorm2d(near_in_channel[1]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[1], out_channel=near_in_channel[1], kernel_size=kernel_size, stride=(1,1), padding=(0,0))
                                     )
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[1], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(0, 0))
                                     )
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[1], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(0, 0))
                                     )
        self.conv2_4 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[2], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(0, 0))
                                     )       
        self.conv2_5 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[2], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(0, 0))
                                     )       
        self.far_conv2_1 = nn.Sequential(nn.Conv2d(in_channels=far_in_channel[0], out_channels=far_in_channel[1], kernel_size=kernel_size, stride=stride),
                                         nn.BatchNorm2d(far_in_channel[1]),
                                         nn.ELU(),
                                         Residual(in_channel=far_in_channel[1], out_channel=far_in_channel[1], kernel_size=kernel_size, stride=(1, 1), padding=(0, 0)))   
        
        self.far_conv2_2 = nn.Sequential(nn.Conv2d(in_channels=far_in_channel[1], out_channels=far_in_channel[2], kernel_size=kernel_size, stride=stride),
                                         nn.BatchNorm2d(far_in_channel[2]),
                                         nn.ELU(),
                                         Residual(in_channel=far_in_channel[2], out_channel=far_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(0, 0)))   
        

if __name__ == "__main__":
    sig = torch.randn(2, 3, 4, 5)
    pixel = PixelConv(in_ch=3, up_scaler=2, kernel_size=(4,3), stride=(1,1), padding=(2,0))
    out = pixel(sig,2)
    print('sc')