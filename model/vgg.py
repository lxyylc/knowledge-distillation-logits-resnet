import torch
import torch.nn as nn
import torch.nn.functional as F

cfg={16:[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512]}

class vgg(nn.Module):
    def __init__(self,depth=16,inChannel=3,outChannel=10):
        super(vgg, self).__init__()
        self.inChannel=inChannel
        self.outChannel=outChannel
        self.my_cfg=cfg[depth]
        self.feature=self.make_layer()
        self.fc=nn.Linear(512,outChannel)


    def make_layer(self):
        layers=[]
        in_c=3
        for m in self.my_cfg:
            if m =='M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers.append(nn.Conv2d(in_c,m,kernel_size=3,stride=1,padding=1))
                layers.append(nn.BatchNorm2d(m))
                layers.append(nn.ReLU(inplace=True))
                in_c=m
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.feature(x)
        out=F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


