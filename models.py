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
    # pylint: disable=too-many-instance-attributes
    # Eleven seems reasonable in this case.
    def __init__(self, *, input_size, hidden_size, num_layers,
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


class TransformerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads

        self.values = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.keys = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.queries = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.fc = nn.Linear(self.dim_head*self.num_heads, hidden_size)

    def forward(self, values, keys, queries, mask):
        N, query_len, _ = queries.shape
        value_len, key_len = values.shape[1], keys.shape[1]

        # 1. Split hidden size into several heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.dim_head)
        keys = keys.reshape(N, key_len, self.num_heads, self.dim_head)
        queries = queries.reshape(N, query_len, self.num_heads, self.dim_head)

        # 2. Linear Layer
        values = self.values(values)
        keys = self.values(keys)
        queries = self.queries(queries)

        # 3. Attention Layer
        # 3.1. Get attention weight by inner product and softmax
        attention = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("1e-30"))
        attention = torch.softmax(attention, dim=3)
        # attention shape: (N, num_heads. query_len, key_len)

        # 3.2. Get context by weighted sum
        contexts = torch.einsum("nhqa, nahd -> nqhd", [attention, values])
        # above einsum internally calulates like
        # 3.2.1. Reshape nahd into nhad
        # 3.3.2. qa * ad, get nhqd
        # 3.2.3. Reshape nhqd into nqhd
        contexts = contexts.reshape(N, query_len, self.num_heads*self.dim_head)
        # contexts shape: (N, query_len, hidden_size)

        # 3.3. Linear Layer
        out = self.fc(contexts)
        return out


class AttentionNormBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_ratio, extend_dimension):
        super().__init__()
        self.attention = TransformerAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, extend_dimension*hidden_size),
            nn.ReLU(),
            nn.Linear(extend_dimension*hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self.norm2 = nn.LayerNorm(hidden_size)
    def forward(self, values, keys, queries, mask):
        out = self.attention(values, keys, queries, mask)
        out_attention = self.dropout(self.norm1(out + queries))
        out_fc = self.fc_block(out_attention)
        out = self.dropout(self.norm2(out_fc + out_attention))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, device,
                 extend_dimension, dropout_ratio, max_sequence):
                super().__init__()
                self.device = device
                self.positional_enc = nn.Embedding(max_sequence, input_size)
                self.fc = nn.Linear(input_size, hidden_size)
                self.attention_norm_block = AttentionNormBlock(hidden_size, num_heads, dropout_ratio, extend_dimension)
                self.dropout = nn.Dropout(dropout_ratio)
    def forward(self, x):
        N, T, _ = x.shape
        positions = torch.arange(0, T).expand(N, T).to(self.device)
        out = self.dropout(x+self.positional_enc(positions))
        out = self.fc(out)
        out = self.attention_norm_block(values=out, keys=out, queries=out, mask=None)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, extend_dimension, dropout_ratio):
        super().__init__()
        self.attention = TransformerAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        self.attention_norm_block = AttentionNormBlock(hidden_size, num_heads, dropout_ratio, extend_dimension)
        self.dropout = nn.Dropout(dropout_ratio)
    def forward(self, x, values, keys, mask):
        attention = self.attention(values=x, keys=x, queries=x, mask=mask)
        queries = self.dropout(self.norm(attention + x))
        out = self.attention_norm_block(values, keys, queries, mask=None)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, extend_dimenstion, num_heads,
                 dropout_ratio, device, max_sequence, output_size):
        super().__init__()
        self.device = device
        self.positional_enc = nn.Embedding(max_sequence, input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.decoder_block = DecoderBlock(hidden_size, num_heads, extend_dimenstion, dropout_ratio)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_ratio)
    def forward(self, y, enc_out, mask):
        N, T, _ = y.shape
        positions = torch.arange(0, T).expand(N, T).to(self.device)
        y = self.dropout(y + self.positional_enc(positions))
        out = self.fc1(y)
        out = self.decoder_block(x=out, values=enc_out, keys=enc_out, mask=mask)
        out = self.fc2(out)
        return out


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, extend_dimension, num_heads,
                 dropout_ratio, device, max_sequece, output_size):
        super().__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size, num_heads, device, extend_dimension, dropout_ratio, max_sequece)
        self.decoder = TransformerDecoder(input_size, hidden_size, extend_dimension, num_heads, dropout_ratio, device, max_sequece, output_size)
        self.device = device
    def make_mask(self, y):
        N, T, _ = y.shape
        mask = torch.tril(torch.ones((T, T))).expand(N, 1, T, T)
        return mask.to(self.device)
    def forward(self, x, y):
        mask = self.make_mask(y)
        enc_out = self.encoder(x)
        dec_out = self.decoder(y, enc_out, mask)
        return dec_out
