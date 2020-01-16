import h5pickle as h5py
import numpy as np
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


class MultihopHdf5Dataset(data.Dataset):
    def __init__(self, file_path, max_q_len, max_doc_len):
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len
        self.fp = h5py.File(file_path, 'r')

    @staticmethod
    def _pad_to(x, n):
        return np.pad(x, (0, n - len(x)), constant_values=0)

    @staticmethod
    def _get_mask(ids, max_len):
        mask = np.zeros((max_len, 1), dtype=np.float32)
        for i in range(len(ids)):
            mask[i, 0] = 1

        return mask


class MultihopTrainset(MultihopHdf5Dataset):
    def __init__(self, file_path, neg_examples, max_q_len=20, max_doc_len=150):
        super().__init__(file_path, max_q_len, max_doc_len)
        fp = self.fp
        self.neg_examples = neg_examples

        self.queries = fp['query']
        self.pos_docs = fp['pos_doc']
        self.neg_docs = fp['neg_docs']

    def __getitem__(self, index):
        query = LongTensor(self.queries[index])
        pos_doc = LongTensor(self.pos_docs[index])
        neg_docs = [LongTensor(self.neg_docs[str(i)][index][:self.max_doc_len]) for i in range(self.neg_examples)]

        return query[:self.max_q_len], pos_doc[:self.max_doc_len], neg_docs

    def __len__(self):
        return len(self.queries)

    @staticmethod
    def collate(batch):
        query_batch, pos_doc_batch, neg_docs_batch = zip(*batch)
        query_lens = LongTensor([len(q) for q in query_batch])
        pos_lens = LongTensor([len(d) for d in pos_doc_batch])

        num_neg_examples = len(neg_docs_batch[0])
        batch_size = len(batch)
        neg_batch_lens = [LongTensor([len(neg_docs_batch[j][i]) for j in range(batch_size)]) for i in
                          range(num_neg_examples)]

        query_batch = pad_sequence(query_batch, batch_first=True, padding_value=0)
        pos_doc_batch = pad_sequence(pos_doc_batch, batch_first=True, padding_value=0)

        neg_padded_batches = []

        for i in range(num_neg_examples):
            cur_batch = []
            for seq in neg_docs_batch:
                cur_batch.append(seq[i])
            padded_batch = pad_sequence(cur_batch, batch_first=True, padding_value=0)
            neg_padded_batches.append(padded_batch)

        return query_batch, query_lens, pos_doc_batch, pos_lens, neg_padded_batches, neg_batch_lens


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
