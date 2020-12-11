import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_model(nn.Module):
    def __init__(self):
        super(LeNet5_model, self).__init__()

        """
        # input img : [3,42,42] or [3, 780, 426] or [3,78,42]
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,  
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        """

        self.layer = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )#CNN convolution, pooling
        self.fc_layer= nn.Sequential(
            nn.Linear(16*9*18, 120),
            nn.ReLU(),
            nn.Dropout(p=0.1), #dropout 적용안하려면 빼면된다
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(84, 10)
        )#그 뒤에 일반


        # initialization function, first checks the module type,
        # then applies the desired changes to the weights

        def init_normal(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight) #He initialization

            #그냥 0.01로 하려면 conv1.weight.data.fill_(0.01)

            #use the modules apply function to recursively apply the initialization

        self.layer.apply(init_normal)
        self.fc_layer.apply(init_normal)


    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 16*9*18)#[16,9,9],[16*105*193][16*9*18]
        out = self.fc_layer(out)
        return out


class Config():
    def __init__(self):
        self.batch_size = 20
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-04
        self.epoch = 43 #데이터가 862개
        self.data_augmentation = False #data argumention 할경우 True
