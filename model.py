#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code taken from https://github.com/namkhanhtran/nn4nqa
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import vocab

from data_source import MultihopTrainset, MultihopTestset
from qa_utils.lightning import BaseRanker


class MultiHopAttentionRanker(BaseRanker):
    """"""

    def __init__(self, vocab_size, embed_size, hidden_size, lr, loss_margin, id_to_word, glove_cache, train_ds, val_ds,
                 test_ds,
                 batch_size, num_neg_examples, bidirectional=False, pooling='max', num_steps=2,
                 att_method='sequential'):

        train_ds = MultihopTrainset(train_ds, num_neg_examples)
        val_ds = MultihopTestset(val_ds)
        test_ds = MultihopTestset(test_ds)
        super().__init__(train_ds, val_ds, test_ds, batch_size)

        self.encoder = Encoder(vocab_size=vocab_size,
                               embed_size=embed_size,
                               hidden_size=hidden_size,
                               id_to_word=id_to_word,
                               glove_cache=glove_cache,
                               bidirectional=bidirectional)

        self.num_steps = num_steps
        self.pooling = pooling  # [raw, max, last, mean]

        self.att_size = hidden_size
        if att_method == 'sequential':
            self.att_layer = SequentialAttention(hidden_size=self.att_size)
        elif att_method == 'mlp':
            self.att_layer = MLPttentionLayer(hidden_size=self.att_size)
        else:
            self.att_layer = BilinearAttentionLayer(hidden_size=self.att_size)

        self.qatt_layer = MultiHopAttention(hidden_size=self.att_size, num_steps=num_steps)

        self.sim_layer = nn.CosineSimilarity(dim=1, eps=1e-8)

        self.lr = lr
        self.loss_margin = loss_margin
        self.save_hyperparameters()

    def single_forward(self, q_batch, q_batch_length, pooling='max'):
        q_batch_length, q_perm_idx = q_batch_length.sort(0, descending=True)
        q_batch = q_batch[q_perm_idx]

        q_out = self.encoder(q_batch, q_batch_length, pooling=pooling)
        if pooling == 'raw':
            q_out = q_out.permute(1, 0, 2)  # len x batch x h --> batch x len x h

        q_inverse_idx = torch.zeros(q_perm_idx.size()[0]).long()
        for i in range(q_perm_idx.size()[0]):
            q_inverse_idx[q_perm_idx[i]] = i

        q_out = q_out[q_inverse_idx]

        return q_out

    def forward(self, inputs):
        q_batch, q_batch_length, doc_batch, doc_batch_length = inputs
        q_out_raw = self.single_forward(q_batch, q_batch_length, pooling='raw')  # b x l x h

        q_out = self.qatt_layer(q_out_raw)
        self.num_steps = 2

        d_out = self.single_forward(doc_batch, doc_batch_length, pooling='raw')

        sim = 0
        for idx in range(self.num_steps + 1):
            d_att = self.att_layer([d_out, q_out[idx]])
            s = self.sim_layer(d_att, q_out[idx])
            sim += s

        return sim.unsqueeze(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        pos_inputs, neg_inputs = batch
        pos_outputs, neg_outputs = torch.sigmoid(self(pos_inputs)), torch.sigmoid(self(neg_inputs))
        loss = torch.mean(torch.clamp(self.loss_margin - pos_outputs + neg_outputs, min=0))
        return {'loss': loss, 'log': {'train_loss': loss}}

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('VOCAB_FILE', help='json file containing the mapping from ids to words, used for glove.')
        parser.add_argument('--hidden_dim', type=int, default=512,
                            help='The hidden dimension used throughout the whole network.')
        parser.add_argument('--embed_dim', type=int, default=300, help='The dimensionality of the GloVe embeddings.')
        parser.add_argument('--glove_cache', default='glove_cache', help='Glove cache directory.')

        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
        parser.add_argument('--loss_margin', type=float, default=0.2, help='Hinge loss margin')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--num_neg_examples', type=int, default=32, help='negative examples')

        return parser


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, id_to_word, glove_cache, bidirectional=False):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.birnn = bidirectional
        if bidirectional:
            self.n_cells = 2
        else:
            self.n_cells = 1

        self.embedding = GloveEmbedding(id_to_word, name='840B', dim=embed_size, cache=glove_cache, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=self.birnn, num_layers=1, dropout=.3)

    def forward(self, inputs, input_length, pooling='max'):
        """"""
        embedded = self.embedding(inputs)  # batch x seq x dim
        embedded = embedded.permute(1, 0, 2)  # seq x batch x dim

        batch_size = embedded.size()[1]
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = Variable(embedded.data.new(*state_shape).zero_())

        packed_input = pack_padded_sequence(embedded, input_length.cpu().numpy())
        self.encoder.flatten_parameters()
        packed_output, (ht, ct) = self.encoder(packed_input, (h0, c0))
        outputs, _ = pad_packed_sequence(packed_output)

        if pooling == 'raw':
            return outputs  # len x batch x 2*h

        if pooling == 'last':
            return ht[-1] if not self.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        outputs = outputs.permute(1, 0, 2)
        if pooling == 'max':
            return torch.max(outputs, 1)[0]  # return values and index --> first values
        else:
            return torch.mean(outputs, 1)


class SequentialAttention(nn.Module):
    """"""

    def __init__(self, hidden_size):
        """"""
        super(SequentialAttention, self).__init__()

        self.hidden_size = hidden_size // 2

        self.encoder = nn.LSTM(hidden_size, self.hidden_size, bidirectional=True, num_layers=1, dropout=.2)

    def forward(self, inputs, pooling='sum'):
        """
        """
        y = inputs[0] * inputs[1].unsqueeze(1).expand_as(inputs[0])  # b x l x h

        y = y.permute(1, 0, 2)  # l x b x h

        batch_size = y.size()[1]
        state_shape = 2, batch_size, self.hidden_size
        h0 = c0 = Variable(y.data.new(*state_shape).zero_())
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(y, (h0, c0))

        outputs = outputs.permute(1, 0, 2)  # len x batch x h --> batch x len x h

        outputs = torch.sum(outputs, 2).unsqueeze(2)  # b x l x 1

        alpha = torch.softmax(outputs, dim=1)

        return torch.sum(inputs[0] * alpha, 1)


class MLPttentionLayer(nn.Module):
    """
    """

    def __init__(self, hidden_size, activation='tanh'):
        super(MLPttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.W_0 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_b = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W_0.data.uniform_(-stdv, stdv)
        self.W_1.data.uniform_(-stdv, stdv)
        self.W_b.data.uniform_(-stdv, stdv)

    def forward(self, inputs, pooling='sum'):
        """
        """
        M = torch.bmm(inputs[0], self.W_0.unsqueeze(0).expand(inputs[0].size(0), *self.W_0.size()))  # batch x len x h
        M += torch.mm(inputs[1], self.W_1).unsqueeze(1).expand_as(M)  # batch x h --> batch x len x h

        M = self.activation(M)  # batch x len x h

        U = torch.bmm(M, self.W_b.unsqueeze(0).expand(M.size(0), *self.W_b.size()))
        alpha = torch.softmax(U, dim=1)  # batch x len x 1

        if pooling == 'max':
            return torch.max(inputs[0] * alpha, 1)[0]  # batch x h
        elif pooling == 'mean':
            return torch.mean(inputs[0] * alpha, 1)
        elif pooling == 'raw':
            return inputs[0] * alpha
        else:
            return torch.sum(inputs[0] * alpha, 1)


class BilinearAttentionLayer(nn.Module):
    """"""

    def __init__(self, hidden_size):
        super(BilinearAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, inputs, pooling='sum'):
        """
        """
        M = torch.mm(inputs[1], self.W).unsqueeze(1).expand_as(inputs[0])  # batch x len x h
        alpha = torch.softmax(torch.sum(M * inputs[0], dim=2), dim=1)  # batch x len
        alpha = alpha.unsqueeze(2).expand_as(inputs[0])

        if pooling == 'max':
            return torch.max(inputs[0] * alpha, 1)[0]  # batch x h
        elif pooling == 'mean':
            return torch.mean(inputs[0] * alpha, 1)
        elif pooling == 'raw':
            return inputs[0] * alpha
        else:
            return torch.sum(inputs[0] * alpha, 1)


class MultiHopAttention(nn.Module):
    """"""

    def __init__(self, hidden_size, num_steps=2):
        """"""
        super(MultiHopAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_steps = num_steps  # <=3 in this implementation

        self.W_u_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_m_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_h_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

        self.W_u_2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_m_2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_h_2 = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

        self.W_u_3 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_m_3 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_h_3 = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W_u_1.data.uniform_(-stdv, stdv)
        self.W_u_m_1.data.uniform_(-stdv, stdv)
        self.W_u_h_1.data.uniform_(-stdv, stdv)
        self.W_u_2.data.uniform_(-stdv, stdv)
        self.W_u_m_2.data.uniform_(-stdv, stdv)
        self.W_u_h_2.data.uniform_(-stdv, stdv)
        self.W_u_3.data.uniform_(-stdv, stdv)
        self.W_u_m_3.data.uniform_(-stdv, stdv)
        self.W_u_h_3.data.uniform_(-stdv, stdv)

    def forward(self, inputs, qvector=None, pooling='sum'):
        """
        """
        m_u = [None] * self.num_steps
        m_u[0] = torch.mean(inputs, 1) * qvector if qvector is not None else torch.mean(inputs, 1)
        u_att = [None] * (self.num_steps + 1)
        u_att[0] = torch.mean(inputs, 1)

        W_unsq = self.W_u_1.unsqueeze(0).expand(inputs.size(0), *self.W_u_1.size())
        M = torch.bmm(inputs, W_unsq)  # b x l x h
        M = torch.tanh(M)
        M = M * torch.tanh(torch.mm(m_u[0], self.W_u_m_1)).unsqueeze(1).expand_as(M)
        U = torch.bmm(M, self.W_u_h_1.unsqueeze(0).expand(M.size(0), *self.W_u_h_1.size()))
        alpha = torch.softmax(U, dim=1)  # batch x len x 1

        u_att[1] = torch.sum(inputs * alpha, 1)

        if self.num_steps > 1:
            m_u[1] = m_u[0] + u_att[0] * qvector if qvector is not None else m_u[0] + u_att[0]
            M = torch.bmm(inputs, self.W_u_2.unsqueeze(0).expand(inputs.size(0), *self.W_u_2.size()))  # b x l x h
            M = torch.tanh(M)
            M = M * torch.tanh(torch.mm(m_u[1], self.W_u_m_2)).unsqueeze(1).expand_as(M)
            U = torch.bmm(M, self.W_u_h_2.unsqueeze(0).expand(M.size(0), *self.W_u_h_2.size()))
            alpha = torch.softmax(U, dim=1)  # batch x len x 1

            u_att[2] = torch.sum(inputs * alpha, 1)

        if self.num_steps > 2:
            m_u[2] = m_u[1] + u_att[1] * qvector if qvector is not None else m_u[1] + u_att[1]
            M = torch.bmm(inputs, self.W_u_3.unsqueeze(0).expand(inputs.size(0), *self.W_u_3.size()))  # b x l x h
            M = torch.tanh(M)
            M = M * torch.tanh(torch.mm(m_u[2], self.W_u_m_3)).unsqueeze(1).expand_as(M)
            U = torch.bmm(M, self.W_u_h_3.unsqueeze(0).expand(M.size(0), *self.W_u_h_3.size()))
            alpha = torch.softmax(U, dim=1)  # batch x len x 1
            u_att[3] = torch.sum(inputs * alpha, 1)

        return u_att


class GloveEmbedding(torch.nn.Module):
    def __init__(self, id_to_word, name, dim, freeze=False, cache=None, padding_idx=None):
        super().__init__()
        self.id_to_word = id_to_word
        self.name = name
        self.dim = dim
        self.padding_idx = padding_idx
        self.glove = vocab.GloVe(name=name, dim=dim, cache=cache)
        weights = self._get_weights()
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze)

    def _get_weights(self):
        weights = []
        i = 0
        for idx in sorted(self.id_to_word):
            word = self.id_to_word[idx]
            if i == self.padding_idx:
                weights.append(torch.zeros([self.dim]))
                i += 1
                continue
            if word in self.glove.stoi:
                glove_idx = self.glove.stoi[word]
                weights.append(self.glove.vectors[glove_idx])
                i += 1
            else:
                # initialize randomly
                weights.append(torch.zeros([self.dim]).uniform_(-0.25, 0.25))
        print(f'Imported {i} words from GloVe')
        # this converts a list of tensors to a new tensor
        return torch.stack(weights)

    def forward(self, x):
        return self.embedding(x)
