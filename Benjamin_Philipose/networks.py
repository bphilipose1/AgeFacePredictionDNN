import torch.nn as nn
import torch
'''
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        # Define a linear layer with input_size features and 1 output (for regression)
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)
'''
class CNNRegression(nn.Module):
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
class MultimodalNetwork(nn.Module):
    def __init__(self, cnn_model, num_numerical_features):
        super(MultimodalNetwork, self).__init__()
        self.cnn_model = cnn_model
        self.linear_regression = LinearRegression(num_numerical_features, 1)

        self.final_fc = nn.Linear(cnn_model.output_features + 1, 1)

    def forward(self, image, numerical_data):
        image_features = self.cnn_model(image)
        image_features = torch.flatten(image_features, start_dim=1)
        
        numerical_features = self.linear_regression(numerical_data)
        
        # Concatenate the features from both models
        combined_features = torch.cat([image_features, numerical_features], dim=1)
        
        # Generate the final output
        output = self.final_fc(combined_features)
        return output'''