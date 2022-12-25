
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision
import time
import os
import copy
import VGG.Model as Model
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

cudnn.benchmark = True

data_transforms = {
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    "val":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),    
    }
data_dir= "E:\datasets\dataset"
image_dataset = {x:datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ["train","val"]}
dataloaders= {x: torch.utils.data.DataLoader(image_dataset[x],batch_size= 4,shuffle=True) for x in ["train","val"]}
dataset_size = {x: len(image_dataset[x]) for x in ["train","val"]}
class_name = image_dataset["train"].classes
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(class_name)

def imshow(inp,title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = inp * mean + std
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated

inputs,classes = next(iter(dataloaders["train"]))
out= torchvision.utils.make_grid(inputs)
#imshow(out,title=[class_name[x] for x in classes])
def train_model(model,criterion,optimizer,scheduler,num_epochs=30):
    start_time = time.time()
    best_model_until_now= copy.deepcopy(model.state_dict())
    best_acc=0.0
    for epoch in range(num_epochs):
        print("epoch number {0} is started".format(epoch))
        print("-"*10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs= inputs.to(device)
                labels= labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects +=torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss/ dataset_size[phase]
            epoch_acc = running_corrects.double/dataset_size[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    FinallTime = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
model_ft = Model.VGG_net(3,2)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)