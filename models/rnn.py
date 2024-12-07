import torch
import torch.nn as nn

class ECG_RNN(nn.Module):
    requires_reshape = False
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(ECG_RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # Expected shape: [batch_size, seq_len, input_size]
        out = out[:, -1, :]   # Take the last hidden state
        out = self.fc(out)
        return out
