import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    requires_reshape = True  

    def __init__(self, input_shape):
        super(CNNLSTM, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 5)  # For multi-class classification NOTE change the second thing per class outputs. 

        # Uncomment below for AFIB/AFLUT
        # self.fc2 = nn.Linear(128, 1)  # For binary classification

    def forward(self, x):
        # print("############### DEBUG #############")
        # print(x.shape[1]) <-- little bugger causing an issue, sent in the wrong dims x-x
        if x.shape[1] != 1: 
            x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels=1, sequence_length)
        
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)

        # Prepare for LSTM
        x = x.permute(0, 2, 1) 
        _, (x, _) = self.lstm(x)
        x = x.squeeze(0)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Use softmax for multi-class classification
        x = self.fc2(x)
        return torch.softmax(x, dim=1)  # For multi-class classification

        # Uncomment below for binary class.
        # return torch.sigmoid(x)  # For binary classification
