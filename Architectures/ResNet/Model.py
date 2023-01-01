from xml.dom.minidom import Identified
import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self,inChannel,intermediateChannel,identity_downsample=None,stride=1):
        super().__init__()
        self.expantion = 4
        self.conv1 = nn.Conv2d(inChannel,intermediateChannel,kernel_size=1,stride=1,padding=0,bias= False)
        self.bn1= nn.BatchNorm2d(intermediateChannel)
        self.conv2 = nn.Conv2d(intermediateChannel,intermediateChannel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2= nn.BatchNorm2d(intermediateChannel)
        self.conv3= nn.Conv2d(intermediateChannel,intermediateChannel*self.expantion,kernel_size=1,padding=0,bias=False)
        self.bn3= nn.BatchNorm2d(intermediateChannel*self.expantion)
        self.relu= nn.ReLU()
        self.identity_downsample= identity_downsample
        self.stride= stride
    def forward(self,x):
        identity= x.clone()
        x= self.conv1(x)
        x= self.bn1(x)
        x= self.relu(x)
        x= self.conv2(x)
        x= self.bn2(x)
        x= self.relu(x)
        x= self.conv3(x)
        x= self.bn3(x)
        if self.identity_downsample is not None:
            identity= self.identity_downsample(identity)
        x+= identity
        x= self.relu(x)
        return x
class ResNet(nn.Module):
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.expantion = 4
        self.in_channels= 64
        self.conv1= nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1= nn.BatchNorm2d(64)
        self.relu= nn.ReLU()
        self.maxpool= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1= self._make_layer(block,layers[0],intermediate_channels=64,stride=1)
        self.layer2= self._make_layer(block,layers[1],intermediate_channels=64*2,stride=2)
        self.layer3= self._make_layer(block,layers[2],intermediate_channels=64*4,stride=2)
        self.layer4= self._make_layer(block,layers[3],intermediate_channels=64*8,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 *4,num_classes)
    def forward(self,x):
        x= self.conv1(x)
        x= self.bn1(x)
        x= self.relu(x)
        x= self.maxpool(x)
        x= self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.layer4(x)

        x= self.avgpool(x)
        x= x.reshape(x.shape[0],-1)
        x= self.fc(x)
        return x
    def _make_layer(self,block,num_residual_blocks,intermediate_channels,stride):
        identity_downsample= None
        layers= []
        if stride!=1 or self.in_channels != intermediate_channels*self.expantion:
            identity_downsample= nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels * self.expantion, kernel_size=1,stride= stride, bias= False,),
                                               nn.BatchNorm2d(intermediate_channels * self.expantion),
            )
        layers.append(block(self.in_channels, intermediate_channels,identity_downsample,stride))
        self.in_channels = intermediate_channels * 4

        for i in range( num_residual_blocks - 1 ):
            layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)
def ResNet50(img_channel=3,num_classes=1000):
    return ResNet(block,[3,4,6,3],img_channel,num_classes)
def ResNet101(img_channel=3,num_classes=1000):
    return ResNet(block,[3,4,23,3],img_channel,num_classes)
def ResNet152(img_channel=3,num_classes=1000):
    return ResNet(block,[3,8,36,3],img_channel,num_classes)


def test():
    BATCH_SIZE= 4
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet50(3,1000)
    y = net(torch.randn(BATCH_SIZE, 3, 224, 224)).to(device)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(y.size())
if __name__ == "__main__":
    test()