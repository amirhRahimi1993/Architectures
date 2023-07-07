import torch
import torch.nn as nn

class GoogLeNet(nn.Module):
    def __init__(self,inChannel=3,numClass=1000):
        super(GoogLeNet, self).__init__()
        self.ConvLayer1= nn.Sequential(nn.Conv2d(inChannel,64,kernel_size=(7,7),stride=(2,2),padding= (3,3)),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(64,192,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.ConvLayer2= nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.ConvLayer3= nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.ConvLayer4= nn.Sequential(
            InceptionBlock(832,256,160,320,32,128,128),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(kernel_size=7,stride= 1),
        )
        self.fc= nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024,numClass)
        )
    def forward(self,x):
        x= self.ConvLayer1(x)
        #print(x.size())
        x = self.ConvLayer2(x)
        #print(x.size())
        x = self.ConvLayer3(x)
        #print(x.size())
        x = self.ConvLayer4(x)
        #print(x.size())
        x= x.reshape(x.shape[0],-1)
        #print(x.size())
        return self.fc(x)

class InceptionBlock(nn.Module):
    def __init__(self,inChannel,onexone,reducer3x3,threexthree,reducer5x5,fivexfive,pooler):
        super(InceptionBlock,self).__init__()
        self.oneIncetion = ConvBlock(inChannel,onexone,kernel_size=1)
        self.threeInception = nn.Sequential(
            ConvBlock(inChannel,reducer3x3,kernel_size=1),
            ConvBlock(reducer3x3,threexthree,kernel_size=(3,3),padding=1)
        )
        self.fiveInception= nn.Sequential(
            ConvBlock(inChannel,reducer5x5,kernel_size=1),
            ConvBlock(reducer5x5,fivexfive,kernel_size=5,padding=2)
        )
        self.poolInception= nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            ConvBlock(inChannel,pooler,kernel_size=1)
        )
    def forward(self,x):
        return torch.cat([self.oneIncetion(x),self.threeInception(x),self.fiveInception(x),self.poolInception(x)],1)


class ConvBlock(nn.Module):
    def __init__(self,inChannel,outChannel,**kwargs):
        super(ConvBlock,self).__init__()
        self.conv= nn.Conv2d(inChannel,outChannel,**kwargs)
        self.BatchNorm= nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.BatchNorm(self.conv(x)))

if __name__ == '__main__':
    x= torch.randn(3,3,224,224)
    model = GoogLeNet(3,1000)
    print(model(x).shape)

"""
def X(name,family):
    print(name + family)

Dictionary= {name : "Amir" , family: "Rahimi" }
X(**kwargs)
"""