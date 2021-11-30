import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(torch.nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.zp1 = nn.ZeroPad2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(p=0.35)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout2d(p=0.35)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.zp2 = nn.ZeroPad2d(1)
        self.bn3 = nn.BatchNorm2d(256)

        self.mp1 = nn.MaxPool2d(2)

        self.drop3 = nn.Dropout2d(p=0.35)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.zp3 = nn.ZeroPad2d(1)
        self.bn4 = nn.BatchNorm2d(512)
        self.drop4 = nn.Dropout2d(p=0.35)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(1024)
        self.drop5 = nn.Dropout2d(p=0.35)
        self.conv6 = nn.Conv2d(1024, 2000, kernel_size=3)
        self.zp4 = nn.ZeroPad2d(2)
        self.bn6 = nn.BatchNorm2d(2000)

        self.mp2 = nn.MaxPool2d(2)

        self.drop6 = nn.Dropout2d(p=0.35)

        self.fc1 = nn.Linear(2000 * 6 * 6, 512)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcdrop1 = nn.Dropout(0.5)

        self.dc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512, 10)

        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.zp1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.zp2(x)
        x = self.bn3(x)

        x = self.mp1(x)

        x = self.drop3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.zp3(x)
        x = self.bn4(x)
        x = self.drop4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.drop5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.zp4(x)
        x = self.bn6(x)

        x = self.mp2(x)

        x = self.drop6(x)

        output = x.view(x.size(0), -1)
        output = self.fc1(output)
        x = F.relu(x)
        output = self.fcbn1(output)
        output = self.fcdrop1(output)

        output = self.dc(output) * 0.5
        output = self.fc2(output)
        x = F.relu(x)

        output = self.fc3(output)

        output = self.sm(output)

        return output


class SimpleCNN_M3(nn.Module):
    def __init__(self):
        super(SimpleCNN_M3, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(3, 3)),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=(3, 3)),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(80, 96, kernel_size=(3, 3)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(96, 112, kernel_size=(3, 3)),
            nn.BatchNorm2d(112),
            nn.ReLU()
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(112, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(128, 144, kernel_size=(3, 3)),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(144, 160, kernel_size=(3, 3)),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )
        self.block10 = nn.Sequential(
            nn.Conv2d(160, 176, kernel_size=(3, 3)),
            nn.BatchNorm2d(176),
            nn.ReLU()
        )
        self.fc = nn.Linear(11264, 10)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.bn(x)

        return output


class SimpleCNN_M5(nn.Module):
    def __init__(self):
        super(SimpleCNN_M5, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(5, 5)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(5, 5)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=(5, 5)),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )

        self.fc = nn.Linear(10240, 10)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.bn(x)

        return output


class SimpleCNN_M7(nn.Module):
    def __init__(self):
        super(SimpleCNN_M7, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(7, 7)),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=(7, 7)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(96, 144, kernel_size=(7, 7)),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(144, 192, kernel_size=(7, 7)),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )

        self.fc = nn.Linear(3072, 10)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.bn(x)

        return output
