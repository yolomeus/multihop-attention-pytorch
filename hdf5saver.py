import h5py
import numpy as np

from qa_utils.preprocessing.base_hdf5_saver import BaseHdf5Saver
from qa_utils.preprocessing.dataset import Dataset


class MANHdf5Saver(BaseHdf5Saver):
    """Hdf5 saver implementation for the MAN input format"""

    def __init__(self, dataset, tokenizer, max_vocab_size, vocab_outfile, n_neg_examples, **kwargs):
        super().__init__(dataset, tokenizer, max_vocab_size, vocab_outfile, **kwargs)
        self.n_neg_examples = n_neg_examples

    def _define_trainset(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('query', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('pos_doc', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('neg_docs', shape=(n_out_examples * self.n_neg_examples,), dtype=vlen_int64)

    def _define_candidate_set(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('query', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('doc', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('q_id', shape=(n_out_examples,), dtype=np.dtype('int64'))
        dataset_fp.create_dataset('label', shape=(n_out_examples,), dtype=np.dtype('int64'))

    def _n_out_samples(self, dataset):
        return len(dataset)

    def _save_train_row(self, fp, query, pos_doc, neg_docs, idx):
        fp['query'][idx] = query
        fp['pos_doc'][idx] = pos_doc
        assert len(neg_docs) == self.n_neg_examples, 'the number of negative examples returned by the dataset ' \
                                                     'doesn\'t match num_neg_examples, expected: {0}, ' \
                                                     'received: {1}'.format(self.n_neg_examples, len(neg_docs))
        for i, neg_doc in enumerate(neg_docs):
            fp['neg_docs'][idx * self.n_neg_examples + i] = neg_doc

    def _save_candidate_row(self, fp, q_id, query, doc, label, idx):
        fp['q_id'][idx] = q_id
        fp['query'][idx] = query
        fp['doc'][idx] = doc
        fp['label'][idx] = label
