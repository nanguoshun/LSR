import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, dropout_embedding, dropout_encoder):
        super(Encoder, self).__init__()
        self.wordEmbeddingDim = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size,self.hidden_size, bidirectional=True)
        self.dropout_embedding = nn.Dropout(p = dropout_embedding)
        self.dropout_encoder = nn.Dropout(p = dropout_encoder)

    def forward(self, seq, lens):
        batch_size = seq.shape[0]
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        seq_embd = self.dropout_embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)

        self.encoder.flatten_parameters()

        output, h = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, lens_argsort_argsort)  # B x m x 2l
        #last hidden state
        h = h.permute(1,0,2).contiguous().view(batch_size,1,-1)
        h = torch.index_select(h, 0, lens_argsort_argsort)
        output = self.dropout_encoder(output)
        h = self.dropout_encoder(h)
        return output, h
