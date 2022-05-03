import random

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout_ratio):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 15)
        self.fc3 = nn.Linear(15, num_classes)
        self.bn1 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.bn1(x)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_ratio):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 5)
        self.fc3 = nn.Linear(5, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)

        out, _ = self.rnn(x, (h0,c0))
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = F.relu(self.fc3(out))
        return out


class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout_ratio=0.5, is_bidirectional=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directional = 2 if is_bidirectional else 1
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=is_bidirectional)

        self.dropout1 = nn.Dropout(dropout_ratio)
        self.fc1 = nn.Linear(2 * self.num_directional * hidden_size, 50)
        self.bn1 = nn.BatchNorm1d(50)

        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(50, 25)

        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(10, num_classes)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_directional * self.num_layers,
                         x.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_directional * self.num_layers,
                         x.shape[0], self.hidden_size).to(device)
        hs, _ = self.rnn(x, (h0,c0))

        contexts = get_contexts_by_selfattention(hs, device)
        out = torch.cat((contexts, hs), dim=2)

        out = self.dropout1(F.relu(self.fc1(out)))
        out = out.reshape(x.shape[0], out.shape[2], x.shape[1])
        out = self.bn1(out)

        out = out.reshape(x.shape[0], x.shape[1], out.shape[1])
        out = self.dropout2(F.relu(self.fc2(out)))

        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


def get_contexts_by_selfattention(hs, device):
    N, T, H = hs.shape
    contexts = torch.zeros(N, T, H).to(device)
    for t in range(T):
        h_t = hs[:, t, :].unsqueeze(1)
        h_t = h_t.repeat(1, T, 1)
        attention = (h_t*hs).sum(axis=2)
        attention = F.softmax(attention, dim=1)
        attention = attention.unsqueeze(2)
        attention = attention.repeat(1, 1, H)
        context = (attention*hs).sum(axis=1)
        contexts[:, t, :] = context
    return contexts


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout_ratio)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        encoder_hs, (h_t, c_t)  = self.lstm(x)
        encoder_hs = encoder_hs.reshape(x.shape[0], encoder_hs.shape[2], x.shape[1])
        encoder_hs = self.bn1(encoder_hs)
        encoder_hs = encoder_hs.reshape(x.shape[0], x.shape[1], encoder_hs.shape[1])
        return encoder_hs, h_t, c_t


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_ratio):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers, batch_first=True, dropout=dropout_ratio)
        self.bn1 = nn.BatchNorm1d(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x, h_t, c_t, encoder_hs, device):
        decoder_hs, (h_t, c_t) = self.rnn(x, (h_t, c_t))
        outputs = get_contexts_and_decoderhs_by_attention(encoder_hs, decoder_hs, device)

        outputs = outputs.reshape(x.shape[0], outputs.shape[2], x.shape[1])
        outputs = self.bn1(outputs)
        outputs = outputs.reshape(x.shape[0], x.shape[1], outputs.shape[1])

        predictions = self.fc(outputs)
        return predictions

    def generate(self, h_t, c_t, encoder_hs, x, target_len, device):
        start_x = x[:, 0, :]
        scores = torch.zeros(start_x.shape[0], target_len, 2).to(device)

        for t in range(target_len):
            start_x = start_x.unsqueeze(2)
            start_x = start_x.to(torch.float32)

            decoder_hs, (h_t, c_t) =  self.rnn(start_x, (h_t, c_t))
            output = get_contexts_and_decoderhs_by_attention(encoder_hs, decoder_hs, device)

            output = output.reshape(start_x.shape[0], output.shape[2], start_x.shape[1])
            output = self.bn1(output)
            output = output.reshape(start_x.shape[0], start_x.shape[1], output.shape[1])
            output = self.fc(output)
            predict = output.argmax(axis=2)

            if t+1 == target_len:
                pass
            else:
                start_x = x[:, t+1, :] if random.random() < 0.5 else predict

            output = output.squeeze(1)
            scores[:, t, :] = output
        return scores


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, dec_input, device):
        encoder_hs, h_t, c_t = self.encoder(enc_input)
        outputs = self.decoder(dec_input, h_t, c_t, encoder_hs, device)
        return outputs

    def generate(self, enc_input, dec_input, target_len, device):
        encoder_hs, h_t, c_t = self.encoder.forward(enc_input)
        scores = self.decoder.generate(h_t, c_t, encoder_hs, dec_input, target_len, device)
        return scores


def get_contexts_and_decoderhs_by_attention(encoder_hs, decoder_hs, device):
    N, decoder_T, H = decoder_hs.shape
    encoder_T = encoder_hs.shape[1]
    outputs = torch.zeros(N, decoder_T, 2 * H).to(device)

    for t in range(decoder_T):
        decoder_ht = decoder_hs[:, t, :].unsqueeze(1)
        decoder_ht = decoder_ht.repeat(1, encoder_T, 1)

        attention_weight = (encoder_hs*decoder_ht).sum(axis=2)
        attention_weight = F.softmax(attention_weight, dim=1)
        attention_weight = attention_weight.unsqueeze(2)
        attention_weight = attention_weight.repeat(1, 1, H)

        context_vector = (attention_weight*encoder_hs).sum(axis=1)

        outputs[:, t, :] = torch.cat((context_vector, decoder_hs[:, t, :]), dim=1)
    return outputs
