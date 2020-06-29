import h5py
import torch
from torch import LongTensor, as_tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


class MultihopHdf5Dataset(data.Dataset):
    def __init__(self, file_path, max_q_len, max_doc_len):
        self.file_path = file_path
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len

        with h5py.File(self.file_path, 'r') as fp:
            self.length = len(fp['query'])


class MultihopTrainset(MultihopHdf5Dataset):
    def __init__(self, file_path, neg_examples, max_q_len=20, max_doc_len=150):
        super().__init__(file_path, max_q_len, max_doc_len)
        self.num_neg_examples = neg_examples

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as fp:
            queries = fp['query']
            pos_docs = fp['pos_doc']
            neg_docs = fp['neg_docs']
            query = LongTensor(queries[index])
            pos_doc = LongTensor(pos_docs[index])
            neg_docs = [LongTensor(neg_docs[index * self.num_neg_examples + i][:self.max_doc_len]) for i in
                        range(self.num_neg_examples)]

            return query[:self.max_q_len], pos_doc[:self.max_doc_len], neg_docs

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        queries, pos_docs, neg_doc_batches = zip(*batch)
        neg_doc_batches = list(zip(*neg_doc_batches))

        neg_docs = []
        for batch in neg_doc_batches:
            neg_docs.extend(batch)
        neg_lens = as_tensor([len(doc) for doc in neg_docs])
        neg_docs = pad_sequence(neg_docs, batch_first=True)

        query_lens = as_tensor([len(query) for query in queries])
        queries = pad_sequence(queries, batch_first=True)

        n_negs = len(neg_doc_batches)
        queries = torch.cat([queries for _ in range(n_negs)])
        query_lens = torch.cat([query_lens for _ in range(n_negs)])

        pos_lens = as_tensor([len(pos_doc) for pos_doc in pos_docs])
        pos_docs = pad_sequence(pos_docs, batch_first=True)

        pos_docs = torch.cat([pos_docs for _ in range(n_negs)])
        pos_lens = torch.cat([pos_lens for _ in range(n_negs)])

        pos_inputs = [queries, query_lens, pos_docs, pos_lens]
        neg_inputs = [queries, query_lens, neg_docs, neg_lens]
        return pos_inputs, neg_inputs


class MultihopTestset(MultihopHdf5Dataset):
    def __init__(self, file_path, max_q_len=20, max_doc_len=150):
        super().__init__(file_path, max_q_len, max_doc_len)

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as fp:
            q_ids = fp['q_id']
            queries = fp['query']
            docs = fp['doc']
            labels = fp['label']

            query = LongTensor(queries[index])
            doc = LongTensor(docs[index])
            q_id = q_ids[index]
            label = labels[index]
            return q_id, query[:self.max_q_len], doc[:self.max_doc_len], label

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        q_id_batch, query_batch, doc_batch, label_batch = zip(*batch)
        query_lens = LongTensor([len(q) for q in query_batch])
        doc_lens = LongTensor([len(d) for d in doc_batch])

        query_batch = pad_sequence(query_batch, batch_first=True, padding_value=0)
        doc_batch = pad_sequence(doc_batch, batch_first=True, padding_value=0)

        # we set negative batches and lens to None since we're not training
        return LongTensor(q_id_batch), (query_batch, query_lens, doc_batch, doc_lens), LongTensor(label_batch)
