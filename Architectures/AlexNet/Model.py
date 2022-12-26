import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,inchannel=3,outchannel=1000) -> None:
        super(AlexNet,self).__init__()
        self.alexture=["c_{0}_64_11_4_2".format(inchannel),"r","M","c_64_192_5_1_2","r","M","c_192_384_3_1_1","r","c_384_256_3_1_1","r","c_256_256_3_1_1","r","M"]
        self.convLayer = self.__Three_layerConvBlock()
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,outchannel),
            )
    def __Three_layerConvBlock(self):
        layer = []
        for n in self.alexture:
            if n[0] == "M":
                layer+=[nn.MaxPool2d((3,3),(2,2))]
            elif n[0] == "r":
                layer+=[nn.ReLU(True)]
            else:
                splitor = n.split("_")
                inChannel = int(splitor[1])
                outChannel = int(splitor[2])
                kernel_size =(int(splitor[3]),int(splitor[3]))
                stride = (int(splitor[4]),int(splitor[4]))
                padding = (int(splitor[5]),int(splitor[5]))
                layer +=[
                    nn.Conv2d(inChannel,outChannel,kernel_size,stride,padding)
                    ]
        return nn.Sequential(*layer)
    def forward(self,x):
        output= self.convLayer(x)
        output= self.avg_pool(output)
        output = torch.flatten(output, 1)
        output= self.classifier(output)
        return output 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlexNet(inchannel = 3, outchannel = 1000).to(device)
    print(model)
    BATCH_SIZE = 3
    my_random_input = torch.randn(BATCH_SIZE,3,224,224).to(device)
    assert model(my_random_input).shape == torch.Size([BATCH_SIZE,1000])
    print(model(my_random_input).shape)