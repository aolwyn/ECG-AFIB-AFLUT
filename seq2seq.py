import torch
import torch.nn as nn

class ECGSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ECGSeq2Seq, self).__init__()
        
        # Encoder (LSTM)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder (Fully Connected layer)
        self.fc = nn.Linear(hidden_size, num_classes)  # Output number of classes (heartbeat types)
        
    def forward(self, x):
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last LSTM output to classify (last hidden state)
        output = self.fc(lstm_out[:, -1, :])  # Taking the output of the last time step
        return output