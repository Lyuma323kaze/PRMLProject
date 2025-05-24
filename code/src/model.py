import math
import torch
import torch.nn as nn
import os


# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, nhead=8, num_layers = 4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        self.dim = dim
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you Transformer model here. You can add additional parameters to the function.
        self.pos_encoder = nn.Parameter(torch.randn(1024, 1, dim) * 0.02)   # first parameter is permitted max_sql
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,             # IO dimension
                                                    nhead=nhead,
                                                    dim_feedforward=4 * dim,  # 前馈网络隐藏层维度
                                                    dropout=0.5,
                                                    activation='gelu',
                                                    batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                    num_layers=num_layers)

        ########################################

        self.decoder = nn.Linear(dim, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        #print(input.device)
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        L = embeddings.size(0)
        src_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(input.device.type)
        src = embeddings * math.sqrt(self.dim)
        # position encoding
        # dynamic position encoding length

        output= self.transformer(src=src,
                                    mask=src_mask,
                                    src_key_padding_mask=None)
        ########################################
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), None

class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network
    3) Output linear layer
    """
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        # Construct your RNN model here.
        self.rnn = nn.RNN(input_size=dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=False)
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -init_uniform, init_uniform)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        output, hidden = self.rnn(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden


class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    1) Embedding layer
    2) LSTM network
    3) Output linear layer
    """
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        self.nvoc = nvoc
        ########################################
        # Construct your LSTM model here.
        # all the matrices right multiply for usage

        # structure of the network
        self.cellScale = hidden_size
        self.hidden_size = hidden_size
        self.numlayers = num_layers
        # parameters for gates
        # biases for the gates
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

        # gate parameters by ParameterList
        self.W_layers = nn.ParameterList()  # multiplying x
        self.U_layers = nn.ParameterList()  # multiplying h
        for _ in range(num_layers):
            self.W_layers.append(nn.Parameter(torch.zeros(4 * hidden_size, dim if _ == 0 else hidden_size)))
            self.U_layers.append(nn.Parameter(torch.zeros(4 * hidden_size, hidden_size)))
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        ########################################
        for param in list(self.W_layers) + list(self.U_layers):
            nn.init.xavier_uniform_(param.data)
        for bias in [self.b]:
            nn.init.zeros_(bias.data)
        ########################################

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)
        seq_len = embeddings.size(0)
        batch_size = embeddings.size(1)
        # With embeddings, you can get your output here.
        ########################################
        # initialize cell and hidden
        if hidden is None:
            h_tot = torch.zeros(self.numlayers, batch_size, self.hidden_size, device=input.device)
            c_tot = torch.zeros(self.numlayers, batch_size, self.hidden_size, device=input.device)
        else:
            h_tot, c_tot = hidden
            h_tot = h_tot.detach()
            c_tot = c_tot.detach()

        outputs = []
        for t in range(seq_len):
            x_t = embeddings[t]  # (batch, dim)
            new_h_tot = []
            new_c_tot = []
            for layer in range(self.numlayers):
                # combine all the gates in one matrix, for faster computation
                gates = (torch.mm(x_t, self.W_layers[layer].t()) +
                        torch.mm(h_tot[layer], self.U_layers[layer].t())+
                        self.b.unsqueeze(0))
                i_gate, f_gate, c_ncont, o_gate = gates.chunk(4, 1) # split

                # activation function
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                o_gate = torch.sigmoid(o_gate)
                c_ncont = torch.tanh(c_ncont)

                # update cell
                c_updated = (f_gate * c_tot[layer]) + (i_gate * c_ncont)
                # update hidden
                h_updated = o_gate * torch.tanh(c_updated)

                new_h_tot.append(h_updated)
                new_c_tot.append(c_updated)
                x_t = h_updated  # next layer

            h_tot = torch.stack(new_h_tot)
            c_tot = torch.stack(new_c_tot)
            outputs.append(h_updated)  # last output only

        output_ = torch.stack(outputs)  # (seq_len, batch, hidden)
        ########################################
        # Output has the dimension of
        # sequence_length * batch_size * number of classes

        output_ = self.drop(output_)
        decoded = self.decoder(output_.view(-1, output_.size(2)))
        return decoded.view(output_.size(0), output_.size(1), decoded.size(-1)), hidden
