from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

num_layers = 2  # number of gru layers


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, feat_nb, num_layers=num_layers):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = True

        self.lstm_encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.linear = nn.Linear(hidden_size, feat_nb)
        self.relu = nn.ReLU()

        #initialize weights
        nn.init.xavier_uniform_(self.lstm_encoder.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm_encoder.weight_hh_l0, gain=np.sqrt(2))


    def forward(self, input):
        encoded_input, ht = self.lstm_encoder(input)
        hl = ht[-1]
        hl = self.relu(hl)
        ht_flatten = hl.view(-1, self.hidden_size)
        ht1 = self.linear(ht_flatten)
        ht1_l2 = F.normalize(ht1, p=2, dim=1)   # we apply l2-norm
        return encoded_input, ht1_l2, input

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, feat_nb, num_layers=num_layers):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = True

        self.linear = nn.Linear(feat_nb, hidden_size)
        self.relu = nn.ReLU()
        self.lstm_decoder = nn.GRU(hidden_size, output_size, num_layers, batch_first=True, dropout=0.4)
        # self.sigmoid = nn.Sigmoid()


        #initialize weights
        nn.init.xavier_uniform_(self.lstm_decoder.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm_decoder.weight_hh_l0, gain=np.sqrt(2))


    def forward(self, encoded_output):
        encoded, ht_flatten, encoder_input = encoded_output
        # We create padding mask, so the training is performed correctly. See explanations in the article.
        # This mask corresponds to the flipped zero-padding applied to the initial sequences.
        mask = encoder_input.clone().detach()
        mask[mask > 0] = 1
        mask = torch.sum(mask, dim=2)
        mask[mask >= 1] = 1
        #we try to add padding to hidden state elements
        last_hidden = self.relu(self.linear(ht_flatten))
        # We clone last hidden state t times, where t is max batch sequence length
        last_hidden = last_hidden.view(last_hidden.size(0), 1, last_hidden.size(1)).repeat([1, encoded.size(1), 1])
        # We transform mask so it has the same shape as the hidden state
        mask = mask.view(mask.size(0), mask.size(1), 1).repeat([1, 1, last_hidden.size(2)])
        idx = [i for i in range(mask.size(1) - 1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        mask = torch.index_select(mask, 1, idx)
        # We apply mask to the hidden state
        last_hidden = last_hidden.mul_(mask.detach())
        decoded_output, ht = self.lstm_decoder(last_hidden)
        # decoded_output = self.sigmoid(decoded_output)
        return decoded_output


class GRUAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, input_size, num_layers)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output