import torch
import torch.nn as nn
import torch.nn.functional as F

class ENmodel(nn.Module):

    def __init__(self, inp_dim, out_dim, stride=1):
        super(ENmodel, self).__init__()
        self.conv1 = nn.Conv2d(inp_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or inp_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, z_dim, block=ENmodel, num_blocks = [2,2,2,2], num_classes=10):
        super(Encoder, self).__init__()
        self.inp_dim = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.build_block(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.build_block(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.build_block(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.build_block(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def build_block(self, block, out_dim, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inp_dim, out_dim, stride))
            self.inp_dim = out_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class Classifier(nn.Module):
    # def __init__(self, encoded_dim, classes = 10):
    #     super(Classifier, self).__init__()
    #     self.fc = nn.Linear(encoded_dim, 256)
    #     self.dropout = nn.Dropout(p=0.2)
    #     self.fc2 = nn.Linear(256, 256)
    #     self.fc3 = nn.Linear(256, classes)
    #     self.softmax = nn.Softmax(dim=1)
    
    # def forward(self, X):
    #     layer1_output = self.fc(X)
    #     layer1_output = self.dropout(layer1_output)
    #     layer1_output = self.fc2(layer1_output)
    #     layer1_output = F.relu(layer1_output)
    #     layer1_output = self.fc3(layer1_output)
    #     layer1_output = self.softmax(layer1_output)
    #     return layer1_output
    def __init__(self, encoded_dim, classes = 10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(encoded_dim, classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        layer1_output = self.fc(X)
        layer1_output = self.softmax(layer1_output)
        return layer1_output



# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Encoder(nn.Module):
#     def __init__(self, d_out=128):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
#         self.fc1 = nn.Linear(64 * 8 * 8, 1024)
#         self.fc2 = nn.Linear(1024, 512)
        
#         self.out = nn.Linear(512, d_out)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = torch.flatten(x, 1)  
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.out(x)
#         return x

# class Classifier(nn.Module):
#     def __init__(self, d_in, num_classes):
#         super(Classifier, self).__init__()
#         self.fc = nn.Linear(d_in, num_classes)

#     def forward(self, x):
#         x = self.fc(x)
#         return x
