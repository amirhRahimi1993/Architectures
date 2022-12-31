from turtle import forward
import torch
import torch.nn as nn

class GoogleNet(nn.Module):
    def __init__(self,aux_logit=True,numClass = 1000):
        super(GoogleNet,self).__init__()
        assert aux_logit == True or aux_logit == False
        self.aux_logit = aux_logit
        self.conv1 = conv_block(
            inChannel=3,
            outChannel=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.maxpool1= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = conv_block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2 =nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.Inception3a= Inception_block(192,64,96,128,16,32,32)
        self.Inception3b= Inception_block(256,128,128,192,32,96,64)
        self.maxpool3= nn.MaxPool2d(kernel_size=3, stride=2, padding= 1)

        self.Inception4a= Inception_block(480,192,96,208,16,48,64)
        self.Inception4b= Inception_block(512,160,112,224,24,64,64)
        self.Inception4c= Inception_block(512,128,128,256,24,64,64)
        self.Inception4d= Inception_block(512,112,144,288,32,64,64)
        self.Inception4e= Inception_block(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.Inception5a= Inception_block(832,256,160,320,32,128,128)
        self.Inception5b= Inception_block(832,384,192,384,48,128,128)

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout= nn.Dropout(p=0.4)
        self.fc1= nn.Linear(1024,numClass)

        if self.aux_logit:
            self.aux1= InceptionAux(512,numClass)
            self.aux2= InceptionAux(528,numClass)
        else:
            self.aux1= self.aux2=None
    def forward(self,x):
        x= self.conv1(x)
        x= self.maxpool1(x)
        x= self.conv2(x)
        x= self.maxpool2(x)

        x= self.Inception3a(x)
        x= self.Inception3b(x)
        x= self.maxpool3(x)

        x= self.Inception4a(x)

        #if self.aux_logit:
        #    aux1= self.aux1(x)
        x= self.Inception4b(x)
        x= self.Inception4c(x)
        x= self.Inception4d(x)

        #if self.aux_logit:
        #    aux2= self.aux2(x)
        x= self.Inception4e(x)
        x = self.maxpool4(x)
        x= self.Inception5a(x)
        x= self.Inception5b(x)
        x= self.avgpool(x)
        x= x.reshape(x.shape[0],-1)
        x= self.dropout(x)
        x= self.fc1(x)
        #if self.aux_logit:
        #    return aux1, aux2, x
        #else:
        return x
        

class Inception_block(nn.Module):
    def __init__(self,inChannel,out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_1x1pool):
        super(Inception_block,self).__init__()
        self.branch1 = conv_block(inChannel,out_1x1,kernel_size = 1)
        self.branch2= nn.Sequential(
            conv_block(inChannel,red_3x3,kernel_size = 1),
            conv_block(red_3x3,out_3x3,kernel_size = (3,3), padding=1),
            )
        self.branch3= nn.Sequential(
            conv_block(inChannel,red_5x5,kernel_size = 1),
            conv_block(red_5x5,out_5x5,kernel_size = 5, padding=2),
            )
        self.branch4= nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1, padding=1),
            conv_block(inChannel,out_1x1pool,kernel_size = 1),
            )
    def forward(self,x):
        return torch.concat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)
class InceptionAux(nn.Module):
    def __init__(self,inChannel,numClass):
        super(InceptionAux,self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv = conv_block(inChannel,128,kernel_size=1)
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(p=0.7)
        self.fc1= nn.Linear(2048,1024)
        self.fc2= nn.Linear(1024,numClass)
    def forward(self,x):
        x= self.pool(x)
        x= self.conv(x)
        x= x.reshape(x.shape[0],-1)
        x= self.relu(self.fc1(x))
        x= self.dropout(x)
        x= self.fc1(x)
        x= self.relu(self.fc1(x))
        x= self.dropout(x)
        x= self.fc2(x)
class conv_block(nn.Module):
    def __init__(self,inChannel,outChannel,**kwargs):
        super(conv_block,self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,outChannel,**kwargs)
        self.batchnorm = nn.BatchNorm2d(outChannel)
    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))
if __name__ == "__main__":
    BATCH_SIZE = 5
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    model = GoogleNet(aux_logit=True, numClass=1000)
    print(model(x).shape)
    assert model(x).shape == torch.Size([BATCH_SIZE, 1000])