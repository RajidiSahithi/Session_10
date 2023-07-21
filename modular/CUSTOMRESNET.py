import torch.nn.functional as F
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #drop=0.01
        # Preparation Layer
        self.conv1 = nn.Sequential (
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )  # Number of Parameters = 3*3*3*64=1728
        # Layer 1
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )  # Number of Parameters = 3*3*64*128 = 73728
        self.conv12 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1,bias=False),# Number of Parameters = 3*3*64*128 = 73728
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1,bias=False),# Number of Parameters = 3*3*64*128 = 73728
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )

        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )

        # Layer 3
        self.conv31 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )
        self.conv32 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )

        self.maxpool = nn.MaxPool2d(kernel_size=4,stride=2)

        # Fully connected
        self.fc = nn.Linear(512, 10, bias=True)


    def forward(self, x):
        x = self.conv1(x)

        x = self.conv11(x)
        R1=x
        x = self.conv12(x)
        x=x+R1

        x = self.conv2(x)

        x = self.conv31(x)
        R2=x
        x = self.conv32(x)
        x=x+R2

        x = self.maxpool(x)
        #x = x.randn(512, 1)

# squeeze the tensor to size 512x
        x = x.squeeze(dim=[2, 3])

        #x = x.view(512, 10)

        x = self.fc(x)

        x = x.view(-1, 10)
        return x
        #y = F.log_softmax(x, dim=-1)
        #return  y

def model_summary(model,input_size):
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
    return model,input_size