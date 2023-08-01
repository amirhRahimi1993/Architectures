import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NN(nn.Module):
    def __init__(self, inputNode, hidden_layer_size, numClass):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(inputNode, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size,numClass)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
inputNode = 28 * 28
hidden_layer_size = 50
numClass = 10

lr = 0.001

momentum = 0.9

BatchSize = 64
numEpoch = 20

train_Getter = datasets.MNIST(root="/dataset/", train= True, download= True, transform=transforms.ToTensor())
test_Getter = datasets.MNIST(root="/dataset/", train= False, download= True, transform=transforms.ToTensor())

trainLoader = DataLoader(train_Getter, shuffle= True, batch_size=BatchSize)
testLoader = DataLoader(test_Getter, shuffle= False, batch_size=BatchSize)

model = NN(inputNode=inputNode, hidden_layer_size=hidden_layer_size, numClass=numClass).to(device)
Loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr= lr, momentum=momentum)

for epoch in range(numEpoch):
    numCorrect = 0
    AllNum = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(trainLoader)):
        data = data.to(device)
        target = target.to(device)
        # print(target)
        data = data.reshape(data.shape[0],-1)
        
        score = model(data)
        
        loss = Loss(score, target)
        _, prediction = torch.max(score,1)
        
        numCorrect += (prediction == target).sum()
        AllNum += data.shape[0]
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    print("Accurace in Epoch {0} is {1}".format(epoch, numCorrect/AllNum))
        
        
def check_accuracy(model, testLoader):
    numCorrect = 0
    AllNum = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(testLoader)):
            data = data.to(device)
            target = target.to(device)
            # print(target)
            data = data.reshape(data.shape[0],-1)
            
            score = model(data)
            
            loss = Loss(score, target)
            _, prediction = torch.max(score,1)
            
            numCorrect += (prediction == target).sum()
            AllNum += data.shape[0]
        
        
        
    return numCorrect/AllNum
    
    
print("Test Accurace is {0}".format(check_accuracy(model, testLoader=testLoader)))
    
    