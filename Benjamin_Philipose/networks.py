import torch.nn as nn
import torch

class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        # Define a linear layer with input_size features and 1 output (for regression)
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)

class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adapool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    '''
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = (5, 5), 
            stride = (1, 1),
            padding = (0, 0)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (0,0)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 64, 
            kernel_size = (3, 3), 
            stride = (2, 2), 
            padding = (0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = (5,5),
            stride = (2,2),
            padding = (0,0)
        )
        self.linear1 = nn.Linear(
            in_features=64,
            out_features=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        #reshape for linear layer
        x = x[:,:,0,0] 
        x = self.linear1(x)     

        return x
        '''
    

class MultiModalNetwork(nn.Module):
    def __init__(self, num_numerical_features):
        super(MultiModalNetwork, self).__init__()
        # CNN part (same as your CNNRegression)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Combined part
        self.fc1 = nn.Linear(in_features=128 + num_numerical_features, out_features=64)  # Adjust for combined dimensions
        self.fc2 = nn.Linear(in_features=64, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, image, numerical_data):
        # CNN part
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adapool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Combine with numerical data
        x = torch.cat([x, numerical_data], dim=1)

        # Further processing
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


