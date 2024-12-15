# This file is for models.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter
import pandas as pd
import matplotlib.pyplot as plt 

import os
import wfdb
import pickle
import sys
import glob
from collections import Counter

import pprint

import dataloaders
import beat_utils
import beat_models


class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
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

        # Calculate LSTM input size after Conv layers
        lstm_input_size = input_size // 8  # Adjust if pooling changes
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape input to match Conv1d expectations
        if x.shape[1] != 1:
            x = x.unsqueeze(1)  # Add channel dimension <-- from rhythm detect. might need here.

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
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        _, (x, _) = self.lstm(x)
        x = x.squeeze(0)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1):
        super(Seq2Seq, self).__init__()

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully Connected Layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, labels=None):
        if x.ndimension() == 2:
            x = x.unsqueeze(1) 

        # Encoder forward pass
        encoder_out, (h_n, c_n) = self.encoder_lstm(x)

        # Decoder input - Use the last encoder hidden state for the decoder
        decoder_input = encoder_out[:, -1, :].unsqueeze(1)  # Take last encoder output as decoder input
        decoder_out, _ = self.decoder_lstm(decoder_input)

        # Fully connected layer for classification 
        output = self.fc(decoder_out[:, -1, :])  # Use the last time step of the decoder

        return output
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim  # Correctly initialize the attribute
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure attention layers are on the correct device
        device = x.device
        batch_size, feature_size = x.size()

        # Adjust layer sizes if needed
        if feature_size != self.input_dim:
            self.attention = nn.Sequential(
                nn.Linear(feature_size, feature_size // 2).to(device),
                nn.ReLU(),
                nn.Linear(feature_size // 2, feature_size).to(device),
                nn.Sigmoid()
            )
            self.input_dim = feature_size  # Update input size tracker

        weights = self.attention(x)
        return x * weights


class MultiBranchResAttentionNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiBranchResAttentionNetwork, self).__init__()

        # Define branches
        self.branch1 = nn.Sequential(
            ResidualBlock(1, 32, kernel_size=7),
            ResidualBlock(32, 64, kernel_size=7),
        )

        self.branch2 = nn.Sequential(
            ResidualBlock(1, 32, kernel_size=3),
            ResidualBlock(32, 64, kernel_size=3),
        )

        # Combine branches
        self.combine = nn.Conv1d(128, 64, kernel_size=1)
        self.bn = nn.BatchNorm1d(64)

        # Dynamically calculate feature size after the branches
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Initialize the attention layer with dynamic input size
        attention_input_size = 64  # Final output after combine + pooling
        self.attention = AttentionModule(input_dim=attention_input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(attention_input_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Ensure input shape (batch_size, channels=1, sequence_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Branch processing
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # Concatenate branches
        x = torch.cat((x1, x2), dim=1)
        x = self.combine(x)
        x = self.bn(x)
        x = torch.relu(x)

        # Apply global pooling
        x = self.global_pool(x).squeeze(2)

        # Apply attention
        x = self.attention(x)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


