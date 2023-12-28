import torch.nn as nn
import torch


#ChatGPT4 helped to get the dimensions right
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        # Initial layers (similar to your original design)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # Additional layers for better feature extraction
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)

        # Incorporate Residual Blocks for deeper networks (optional)
        # self.resblock1 = ResidualBlock(512)
        # self.resblock2 = ResidualBlock(512)

        self.adapool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        # x = self.resblock1(x)
        # x = self.resblock2(x)
        x = self.adapool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
#ChatGPT4 helped to get the dimensions right

class MultiModalNetwork(nn.Module):
    def __init__(self, num_numerical_features):
        super(MultiModalNetwork, self).__init__()
        # CNN part
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))

        # combined part
        self.fc1 = nn.Linear(512 + num_numerical_features, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, image, numerical_data):
        # CNN processing
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.adapool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Combine with numerical data
        x = torch.cat([x, numerical_data], dim=1)

        # combined processing
        x = self.dropout1(self.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn_fc2(self.fc2(x))))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x