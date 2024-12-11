import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1):
        super(ECGSeq2Seq, self).__init__()

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Decoder LSTM (This will also output a fixed-size context vector for classification)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully Connected Layer for classification (outputting the number of classes)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Encoder forward pass
        encoder_out, (h_n, c_n) = self.encoder_lstm(x)

        # Decoder forward pass (you can adjust the decoder input as needed; we use the encoder's last hidden state)
        decoder_input = encoder_out[:, -1, :].unsqueeze(1)  # Use last encoder output as decoder input
        decoder_out, _ = self.decoder_lstm(decoder_input)

        # Fully connected layer for classification (use the last time step of the decoder)
        output = self.fc(decoder_out[:, -1, :])  # Take the last time step's output for classification

        return output