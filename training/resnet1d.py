import torch 
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=1),
                                         nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_layers(x)
        if self.downsample:
            x = self.downsample(x)
        
        out += x

        return self.relu(out)
    

class ResNet(nn.Module):
    def __init__(self, block, in_channels, out_size, kernel_size, stride, padding, layers):
        super(ResNet, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, 8, kernel_size=kernel_size, stride=stride, padding=padding),
                                        nn.BatchNorm2d(8),
                                        nn.ReLU())
        
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.layer0 = self._make_layer(block, 8, 8, 3, 1, layers[0], True)
        self.layer1 = self._make_layer(block, 8, 12, 3, 2, layers[1], False)
        self.layer2 = self._make_layer(block, 12, 16, 3, 2, layers[2], False)
        self.layer3 = self._make_layer(block, 16, 24, 3, 2, layers[3], False)

        self.end_pool = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(384, out_size)
    
    def _make_layer(self, block, in_channels, out_channels, kernel, stride, layers, first=True):
        downsample = None
        if not first:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(out_channels))
            
        blocks = [block(in_channels, out_channels, kernel, stride, 1, downsample)]
        for _ in range(1, layers):
            blocks.append(block(out_channels, out_channels, kernel, 1, 1))

        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.pool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.end_pool(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return nn.functional.softmax(x, dim=1)