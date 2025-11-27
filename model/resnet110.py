import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self,inChannel,outChannel,stride=1):
        super(ResidualBlock, self).__init__()
        #有卷积的路径，卷积层-BN-RELU-卷积层-BN
        self.convpath=nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            #此处输入维度是上一个卷积层的输出维度
            nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outChannel)
        )
        #残差路径
        if inChannel!=outChannel or stride!=1:
            #输入输出维度不一样时，用1*1卷积核进行维度变换
            self.shortcut=nn.Sequential(
                nn.Conv2d(inChannel,outChannel,kernel_size=1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(outChannel)
            )
        else:
            self.shortcut=nn.Sequential()

    def forward(self,x):
        #实现残差连接，把两条路的输出相加然后relu
        return F.relu(self.convpath(x)+self.shortcut(x))

class ResNet(nn.Module):
    #cifar10
    def __init__(self,ResidualBlock,num_class=10):
        super(ResNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        #就改了这个num_block
        self.layer1 = self.make_layer(ResidualBlock, 16, 16, 18, stride=1)  # 不降采样
        self.layer2 = self.make_layer(ResidualBlock, 16, 32, 18, stride=2)  # 高宽减半
        self.layer3 = self.make_layer(ResidualBlock, 32, 64, 18, stride=2)  # 高宽再减半
        self.fc=nn.Linear(256,num_class)

    def make_layer(self,block, in_ch, out_ch, num_blocks, stride):
        layers = []
        #第一个残差块，负责降维
        layers.append(block(in_ch, out_ch, stride))
        #剩下的残差块
        for i in range(0,num_blocks-1):
            #这些block之间维度也要对上
            layers.append(block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet110():
    return ResNet(ResidualBlock)



