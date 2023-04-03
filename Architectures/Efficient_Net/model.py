import torch
import torch.nn as nn
from math import ceil
print(torch.__version__)

base_model= [
    #expand_ratio, channels,repeats,stride,kernel_size
    [1,16,1,1,3],
    [6,24,2,2,3],
    [6,40,2,2,5],
    [6,80,3,2,3],
    [6,112,3,1,5],
    [6,192,4,2,5],
    [6,328,1,1,3],
]

phi_values= {
    # tuple of : (phi_value,resulution, drop_rate)
    "b0":(0,224,0.2),#alpha,beta,gamma,depth = alpha ** phi
    "b1":(0.5,240,0.2),
    "b2":(1,260,0.3),
    "b3": (2,300,0.3),
    "b4": (3,380,0.3),
    "b5": (4,456,0.4),
    "b6": (5,528,0.5),
    "b7": (6,600,0.5),
}

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,group=1):
        super(CNNBlock,self).__init__()
        self.cnn= nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=group,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    def forward(self,x):
        return self.silu(self.bn(self.cnn(x)))
class SqueezeExcitation(nn.Module):
    def __init__(self,in_channels,reduce_dim):
        super(SqueezeExcitation,self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels,reduce_dim,1),
            nn.SiLU(),
            nn.Conv2d(reduce_dim,in_channels,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self,
                 inchannels,
                 outchannels,
                 kernelsize,
                 stride,
                 padding,
                 expand_ratio,
                 reduction=4, # squeeze exitation
                 survivial_prob=0.8 # for stochastic depth
                 ):
        
        super(InvertedResidualBlock,self).__init__()
        self.survival_prob= survivial_prob
        self.use_redidual= inchannels == outchannels and stride == 1
        hidden_dim = inchannels * expand_ratio
        self.expand= inchannels != hidden_dim
        reduced_dim = int(inchannels/ reduction)
        if self.expand:
            self.expand_conv = CNNBlock(
                inchannels,hidden_dim,kernel_size=3,stride=1,padding=1
            )
        self.conv= nn.Sequential(
            CNNBlock(hidden_dim,hidden_dim,kernelsize,stride,padding,group=hidden_dim),
            SqueezeExcitation(hidden_dim,reduced_dim),
            nn.Conv2d(hidden_dim,outchannels,1,bias=False),
            nn.BatchNorm2d(outchannels),
        )
    def stochastic_depth(self,x):
        if not self.training:
            return x
        binary_tensor= torch.rand(x.shape[0],1,1,1,device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor
    def forward(self,inputs):
        x= self.expand_conv(inputs) if self.expand else inputs
        if self.use_redidual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
class EfficientNet(nn.Module):
    def __init__(self,version, numclasses) -> None:
        super(EfficientNet,self).__init__()
        width_factor, depth_factor, dropout_rate= self.calculate_factors(version)
        last_channels= ceil(1280 * width_factor)
        self.pool= nn.AdaptiveAvgPool2d(1)
        self.feature= self.create_feautres(width_factor,depth_factor,last_channels)
        self.classifier= nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels,numclasses),
        )
    def calculate_factors(self,version,alpha= 1.2,beta= 1.1):
        phi, res, drop_rate= phi_values[version]
        depth_factor= alpha ** phi
        width_factor= beta ** phi
        return width_factor, depth_factor, drop_rate
    def create_feautres(self,width_factor,depth_factor,last_channels):
        channels= int(32 * width_factor)
        features= [CNNBlock(3,channels,3,stride=2,padding=1)]
        in_channels= channels
        for expand_ratio, channels,repeats,stride,kernel_size in base_model:
            out_channels= 4*ceil(int(channels * width_factor)/4)
            layers_repeats = ceil(repeats * depth_factor)
            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(in_channels,out_channels,expand_ratio=expand_ratio,stride=stride if layer == 0 else 1,kernelsize=kernel_size,padding=kernel_size//2)
                )
                in_channels = out_channels
        features.append(CNNBlock(in_channels,last_channels,kernel_size=1,padding=1,stride=1))
        return nn.Sequential(*features)
    def forward(self,x):
        x= self.pool(self.feature(x))
        return self.classifier(x.view(x.shape[0],-1))
def test():
    device= "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    num_example, num_classes = 4,10
    phi, res, drop_rate= phi_values[version]
    x= torch.randn((num_example,3,res,res))
    model= EfficientNet(version=version,numclasses=num_classes).to(device)
    print(model(x).shape)
test()