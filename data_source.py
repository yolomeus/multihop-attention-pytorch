import h5pickle as h5py
import numpy as np
import torch
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


class MultihopHdf5Dataset(data.Dataset):
    def __init__(self, file_path, max_q_len, max_doc_len):
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len
        self.fp = h5py.File(file_path, 'r')


class MultihopTrainset(MultihopHdf5Dataset):
    def __init__(self, file_path, neg_examples, max_q_len=20, max_doc_len=150):
        super().__init__(file_path, max_q_len, max_doc_len)
        fp = self.fp
        self.num_neg_examples = neg_examples

        self.queries = fp['query']
        self.pos_docs = fp['pos_doc']
        self.neg_docs = fp['neg_docs']

    def __getitem__(self, index):
        query = LongTensor(self.queries[index])
        pos_doc = LongTensor(self.pos_docs[index])
        neg_docs = [LongTensor(self.neg_docs[index * self.num_neg_examples + i][:self.max_doc_len]) for i in
                    range(self.num_neg_examples)]
        neg_docs_lens = [len(doc) for doc in neg_docs]

        return query[:self.max_q_len], pos_doc[:self.max_doc_len], neg_docs, neg_docs_lens

    def __len__(self):
        return len(self.queries)

    def collate(self, batch):
        batch_size = len(batch)
        queries, query_lengths, pos_docs, pos_doc_lengths, neg_docs, neg_doc_lengths = \
            [], [], [], [], [], []

        # in order to pad the sequences, they must be in a flat list first
        for b_query, b_pos_doc, b_neg_docs, b_neg_doc_lengths in batch:
            queries.append(b_query)
            query_lengths.append(len(b_query))
            pos_docs.append(b_pos_doc)
            pos_doc_lengths.append(len(b_pos_doc))
            neg_docs.extend(b_neg_docs)
            neg_doc_lengths.extend(b_neg_doc_lengths)

        queries = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
        query_lengths = torch.LongTensor(query_lengths)
        pos_docs = torch.nn.utils.rnn.pad_sequence(pos_docs, batch_first=True)
        pos_doc_lengths = torch.LongTensor(pos_doc_lengths)
        neg_docs = torch.nn.utils.rnn.pad_sequence(neg_docs, batch_first=True)
        neg_doc_lengths = torch.LongTensor(neg_doc_lengths)

        pos_inputs = [queries, query_lengths, pos_docs, pos_doc_lengths]

        # for the negative inputs, we need to repeat the queries for each negative example and then
        # split everything again
        queries_neg = np.repeat(queries, self.num_neg_examples, axis=0)
        queries_neg = np.split(queries_neg, batch_size)
        query_lengths_neg = np.repeat(query_lengths, self.num_neg_examples, axis=0)
        query_lengths_neg = np.split(query_lengths_neg, batch_size)
        neg_docs = np.split(neg_docs, batch_size)
        neg_doc_lengths = np.split(neg_doc_lengths, batch_size)

        neg_inputs = [queries_neg, query_lengths_neg, neg_docs, neg_doc_lengths]

        return pos_inputs, neg_inputs


class MultihopTestset(MultihopHdf5Dataset):
    def __init__(self, file_path, max_q_len=20, max_doc_len=150):
        super().__init__(file_path, max_q_len, max_doc_len)
        fp = self.fp

        self.q_ids = fp['q_id']
        self.queries = fp['query']
        self.docs = fp['doc']
        self.labels = fp['label']

    def __getitem__(self, index):
        query = LongTensor(self.queries[index])
        doc = LongTensor(self.docs[index])
        q_id = self.q_ids[index]
        label = self.labels[index]
        return q_id, query[:self.max_q_len], doc[:self.max_doc_len], label

    def __len__(self):
        return len(self.queries)

    @staticmethod
    def collate(batch):
        q_id_batch, query_batch, doc_batch, label_batch = zip(*batch)
        query_lens = LongTensor([len(q) for q in query_batch])
        doc_lens = LongTensor([len(d) for d in doc_batch])

        query_batch = pad_sequence(query_batch, batch_first=True, padding_value=0)
        doc_batch = pad_sequence(doc_batch, batch_first=True, padding_value=0)

        # we set negative batches and lens to None since we're not training
        return LongTensor(q_id_batch), (query_batch, query_lens, doc_batch, doc_lens), LongTensor(label_batch)
