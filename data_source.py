import h5py
from torch import LongTensor, as_tensor
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

        return query[:self.max_q_len], pos_doc[:self.max_doc_len], neg_docs

    def __len__(self):
        return len(self.queries)

    def collate(self, batch):
        queries, pos_docs, neg_doc_batches = zip(*batch)

        neg_doc_batches = list(zip(*neg_doc_batches))

        query_lens = as_tensor([len(query) for query in queries])
        queries = pad_sequence(queries, batch_first=True)

        pos_lens = as_tensor([len(pos_doc) for pos_doc in pos_docs])
        pos_docs = pad_sequence(pos_docs, batch_first=True)

        neg_inputs = [[queries, query_lens, pad_sequence(list(neg_batch), batch_first=True), as_tensor([len(neg) for neg in neg_batch])]
                      for neg_batch in neg_doc_batches]

        return [queries, query_lens, pos_docs, pos_lens], neg_inputs


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
