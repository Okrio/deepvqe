
import torch
# import torchaudio
import torch.nn as nn
import numpy as np


class AlignBlock(nn.Module):
    def __init__(self,in_ch=128,h=32, max_delay_block=32) -> None:
        super().__init__()
        # self.p_dim = pdim
        self.h = h
        self.max_delay = max_delay_block
        self.pointwise_conv1 = nn.Conv2d(in_ch,h, kernel_size=(1,1) )
        self.pointwise_conv2 = nn.Conv2d(in_ch, h, kernel_size=(1,1))
        self.conv2d = nn.Conv2d(h, 1, kernel_size=(5,3),stride=(1,1),padding=(3,1))
        self.neg_inf = -10e12

    def forward(self, xm, xf):
        # b,c,t,f
        b,c,t,f = xf.shape
        xm_1 = self.pointwise_conv1(xm)
        xf_1= self.pointwise_conv2(xf)
        padd = torch.zeros((b,self.h,self.max_delay-1, f))
        k = torch.cat((padd, xf_1), dim=2)
        b,c,t,f = k.shape
        k_tmp = k.reshape(b*c, t,f)
        k_1 =  k_tmp.unfold(1, self.max_delay, step=1)
        k_2 = k_1.chunk(b,0)
        k_2 = torch.stack(k_2,dim=0)
        q = xm_1.unsqueeze(-2)
        corr = torch.matmul( q,k_2).squeeze(-2)
        corr_1 = self.conv2d(corr)[...,:-2,:].squeeze(1)

        # mask for previous padd
        min_t = min(corr_1.size(1), self.max_delay)
        mask = torch.ones(min_t, self.max_delay)
        mask = torch.tril(mask).flip(dims=[-1])
        mask = mask.unsqueeze(0).repeat(corr.size(0), 1,1)
        mask = torch.logical_not(mask)
        corr_1[:,:self.max_delay][mask] = self.neg_inf
        corr_2 = torch.softmax(corr_1, dim=-1)

        # weighted sum
        b,c,t,f = xf.shape
        padd = torch.zeros(b,c,self.max_delay-1,f)
        xf_2 = torch.cat((padd, xf),-2).transpose(3,2)
        xf_2 = xf_2.unfold(-1, self.max_delay, step=1)
        corr_ = corr_2.unsqueeze(1).unsqueeze(1)
        out = (corr_ * xf_2).sum(-1).transpose(3,2)
        return out





class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding) -> None:
        super().__init__()
        
        self.padding = padding
        self.resid = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ELU())
        
    def forward(self, x):
        x1 = self.resid(x)
        x1 = x1[...,:-self.padding[0],:-self.padding[1]]
        out = x1 + x
        return out


class PixelConv(nn.Module):
    def __init__(self, in_ch, out_ch, up_scaler, kernel_size, stride, padding) -> None:
        super().__init__()
        self.up_scaler = up_scaler
        self.out_ch = out_ch
        self.padding = [kernel_size[0] - stride[0], kernel_size[1] - stride[1]]
        self.conv2d = nn.Conv2d(in_channels=in_ch, out_channels=out_ch*up_scaler, stride=stride, kernel_size=kernel_size, padding=padding)

        # self.pixel = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, stride=stride, kernel_size=kernel_size, padding=padding),
        #                            nn.PixelShuffle(2*in_ch))
        
    def forward(self, x):
        # b c t f
        b,c,t,f = x.shape
        out1 = self.conv2d(x)
        out1 = out1[...,:-self.padding[0],:-self.padding[1]]
        out2 = torch.chunk(out1, self.up_scaler, dim=1)
        out3 = torch.stack(out2, -1)
        out4 = out3.reshape(b,self.out_ch,t,-1)

        # out = self.pixel(x)
        return out4




class SkipBlock(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(1,1), bias=False)
    def forward(self, x,y):
        x1 = self.conv2(x)
        out = x1 + y
        return out 

class Bottleneck(nn.Module):
    def __init__(self, infeat_num, rnn_hidden, rnn_num) -> None:
        super().__init__()
        self.feat_nums = infeat_num
        self.rnn_hidden = rnn_hidden
        self.gru = nn.GRU(self.feat_nums, self.rnn_hidden, batch_first=True)
        self.linear = nn.Linear(self.rnn_hidden, self.feat_nums)
    
    def forward(self, x):
        x1 = x.transpose(2,1)
        b,t,c,f = x1.shape
        x2 = x1.reshape(b,t,-1)
        gru_out,_ = self.gru(x2)
        lin_out = self.linear(gru_out)
        lin_out_re = lin_out.reshape(b,t,c,f)
        lin_out_re = lin_out_re.transpose(2,1)
        return lin_out_re


class CCMBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v = torch.tensor([1, -0.5+1j*0.5*np.sqrt(3), -0.5 - 1j*0.5*np.sqrt(3)]).unsqueeze(-1)
        self.m = 2
        self.n = 1

    
    def forward(self, x):
        b,c,t,f = x.shape
        x1= torch.chunk(x, 3, 1)
        x2 = torch.stack(x1, dim=-1)
        x3 = torch.matmul( x2.type_as(self.v),self.v).squeeze(-1)
        x4 = x3.reshape(b, self.m+1, self.n*2+1, t,f)

        return 0
    
class deepvqe(nn.Module):
    def __init__(self, near_in_channel=[2, 64, 128],out_ch=27, far_in_channel=[2, 32, 128], kernel_size=[4, 3], stride=[1, 2]) -> None:
        super().__init__()
        self.rnn_infeat = int(np.ceil( 240 / 2**5) * near_in_channel[-1])
        self.padding = [kernel_size[0] - stride[0], kernel_size[1] - stride[1]]
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[0], out_channels=near_in_channel[1], kernel_size=kernel_size, stride=stride, padding=self.padding),
                                     nn.BatchNorm2d(near_in_channel[1]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[1], out_channel=near_in_channel[1], kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1,kernel_size[1]-1))
                                     )
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[1], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride,padding=self.padding),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size[0]-1,kernel_size[1]-1))
                                     )
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[2]*2, out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride,padding=self.padding),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size[0]-1,kernel_size[1]-1))
                                     )
        self.conv2_4 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[2], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride, padding = self.padding),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size[0]-1,kernel_size[1]-1))
                                     )       
        self.conv2_5 = nn.Sequential(nn.Conv2d(in_channels=near_in_channel[2], out_channels=near_in_channel[2], kernel_size=kernel_size, stride=stride, padding=self.padding),
                                     nn.BatchNorm2d(near_in_channel[2]),
                                     nn.ELU(),
                                     Residual(in_channel=near_in_channel[2], out_channel=near_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size[0]-1,kernel_size[1]-1))
                                     )       
        self.far_conv2_1 = nn.Sequential(nn.Conv2d(in_channels=far_in_channel[0], out_channels=far_in_channel[1], kernel_size=kernel_size, stride=stride, padding=self.padding),
                                         nn.BatchNorm2d(far_in_channel[1]),
                                         nn.ELU(),
                                         Residual(in_channel=far_in_channel[1], out_channel=far_in_channel[1], kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size[0]-1,kernel_size[1]-1)))   
        
        self.far_conv2_2 = nn.Sequential(nn.Conv2d(in_channels=far_in_channel[1], out_channels=far_in_channel[2], kernel_size=kernel_size, stride=stride, padding=self.padding),
                                         nn.BatchNorm2d(far_in_channel[2]),
                                         nn.ELU(),
                                         Residual(in_channel=far_in_channel[2], out_channel=far_in_channel[2], kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size[0]-1,kernel_size[1]-1)))   
        self.skip5 = SkipBlock(in_ch=near_in_channel[-1])
        self.skip4 = SkipBlock(in_ch=near_in_channel[-1])
        self.skip3 = SkipBlock(in_ch=near_in_channel[-1])
        self.skip2 = SkipBlock(in_ch=near_in_channel[-1])
        self.skip1 = SkipBlock(in_ch=near_in_channel[1])
        self.deconv2_5 = nn.Sequential(Residual(in_channel=near_in_channel[-1], out_channel=near_in_channel[-1], kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1,kernel_size[1]-1)),
                                       PixelConv(in_ch=near_in_channel[-1],out_ch=near_in_channel[-1], up_scaler=2, kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1, kernel_size[1]-1)),
                                       nn.BatchNorm2d(near_in_channel[-1]),
                                       nn.ELU())
        self.deconv2_4 = nn.Sequential(Residual(in_channel=near_in_channel[-1], out_channel=near_in_channel[-1], kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1,kernel_size[1]-1)),
                                       PixelConv(in_ch=near_in_channel[-1],out_ch=near_in_channel[-1], up_scaler=2, kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1, kernel_size[1]-1)),
                                       nn.BatchNorm2d(near_in_channel[-1]),
                                       nn.ELU())        
        self.deconv2_3 = nn.Sequential(Residual(in_channel=near_in_channel[-1], out_channel=near_in_channel[-1], kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1,kernel_size[1]-1)),
                                       PixelConv(in_ch=near_in_channel[-1],out_ch=near_in_channel[-1], up_scaler=2, kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1, kernel_size[1]-1)),
                                       nn.BatchNorm2d(near_in_channel[-1]),
                                       nn.ELU())
        self.deconv2_2 = nn.Sequential(Residual(in_channel=near_in_channel[-1], out_channel=near_in_channel[-1], kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1,kernel_size[1]-1)),
                                       PixelConv(in_ch=near_in_channel[-1],out_ch=near_in_channel[1], up_scaler=2, kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1, kernel_size[1]-1)),
                                       nn.BatchNorm2d(near_in_channel[1]),
                                       nn.ELU())       
        self.deconv2_1 = nn.Sequential(Residual(in_channel=near_in_channel[1], out_channel=near_in_channel[1], kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1,kernel_size[1]-1)),
                                       PixelConv(in_ch=near_in_channel[1],out_ch=out_ch, up_scaler=2, kernel_size=kernel_size, stride=(1,1), padding=(kernel_size[0]-1, kernel_size[1]-1)))     
        self.bottleneck = Bottleneck(infeat_num=self.rnn_infeat, rnn_hidden=self.rnn_infeat,rnn_num=1)
        self.alignblock = AlignBlock(128, 32, 32)

    def forward(self, x, far=None):
        x1 = self.conv2_1(x)[...,:-self.padding[0],:]
        x2 = self.conv2_2(x1)[...,:-self.padding[0],:]

        far_1 = self.far_conv2_1(far)[...,:-self.padding[0],:]
        far_2 = self.far_conv2_2(far_1)[...,:-self.padding[0],:]
        align_out = self.alignblock(x2,far_2)
        x2_conc = torch.cat((x2, align_out), 1)
        x3 = self.conv2_3(x2_conc)[...,:-self.padding[0],:]
        x4 = self.conv2_4(x3)[...,:-self.padding[0],:]
        x5 = self.conv2_5(x4)[...,:-self.padding[0],:]
        x6 = self.bottleneck(x5)
        
        skip5 = self.skip5(x5,x6)
        x7 = self.deconv2_5(skip5)[...,:-1]
        skip4 = self.skip4(x4, x7)
        x8 = self.deconv2_4(skip4)
        skip3 = self.skip3(x3, x8)
        x9 = self.deconv2_3(skip3)
        skip2 = self.skip2(x2, x9)
        x10 = self.deconv2_2(skip2) 
        skip1 = self.skip1(x1, x10)
        x11 = self.deconv2_1(skip1)
        return x11


if __name__ == "__main__":
    # sig = torch.randn(2,128,3,60)
    # sig1 = torch.randn(2,128,3,60)
    # alignModule = AlignBlock(128)
    # out = alignModule(sig, sig1)

    # sig = torch.randn(2,27,1,4)
    # ccm = CCMBlock()
    # out = ccm(sig)
    sig = torch.randn(2, 2, 1, 240)
    far = torch.randn(2,2,1,240)
    net = deepvqe()
    out3 = net(sig,far)
    # conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4,3), stride=(1,2),padding=(3,1))
    # out2 = conv(sig)
    # residual = Residual(in_channel=3, out_channel=3, kernel_size=(4,3), stride=(1,1), padding=(3,2))
    # out1 = residual(sig)
    # sig1 = torch.randn(2,128,1,8)
    # pixel = PixelConv(in_ch=128, out_ch=64,up_scaler=2, kernel_size=(4,3), stride=(1,1), padding=(3,2))

    # out = pixel(sig1)
    print('sc')