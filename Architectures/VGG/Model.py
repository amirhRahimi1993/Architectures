import torch
import torch.nn as nn

VGG_types = {
    "VGG11":[64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "VGG13":[64,64,"M",128,128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "VGG16":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
    "VGG19":[64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"],
    }
class VGG_net(nn.Module):
    def __init__(self,inchannel=3, numclass= 1000):
        super(VGG_net,self).__init__()
        self.inchannel = inchannel
        self.numclass = numclass
        self.conv_later = self.create_conv_layers(VGG_types["VGG16"])
        self.fcs= nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,self.numclass),
        )
    def forward(self,x):
        x=self.conv_later(x)
        x= x.reshape(x.shape[0],-1)
        x=self.fcs(x)
        return x
    def create_conv_layers(self,architectire):
        layers= []
        inchannel = self.inchannel
        for x in architectire:
            if type(x) == int:
                outchannel = x
                layers+=[
                    nn.Conv2d(
                        in_channels = inchannel,
                        out_channels=outchannel,
                        kernel_size= (3,3),
                        stride = (1,1),
                        padding = (1,1),
                        ),
                    nn.BatchNorm2d(x),# Barch norm orginally indtroduce in 2016 however VGG introduced in 2014
                    nn.ReLU(),                    
                    ]
                inchannel = outchannel
            else:
                layers+=[nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
        return nn.Sequential(*layers)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG_net(inchannel = 3, numclass = 1000).to(device)
    print(model)
    BATCH_SIZE = 3
    my_random_input = torch.randn(BATCH_SIZE,3,224,224).to(device)
    assert model(my_random_input).shape == torch.Size([BATCH_SIZE,1000])
    print(model(my_random_input).shape)