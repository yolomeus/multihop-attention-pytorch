import os

import h5py
import numpy as np

from qa_utils.preprocessing.dataset import Dataset
from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.hdf5saver import Hdf5Saver
from tokenizer import NLTKTokenizer


class MANHdf5Saver(Hdf5Saver):
    def __init__(self, dataset: Dataset, tokenizer, max_vocab_size, vocab_outfile, max_query_len, max_doc_len,
                 train_outfile=None, dev_outfile=None,
                 test_outfile=None):
        super().__init__(dataset, tokenizer, max_vocab_size, vocab_outfile, train_outfile, dev_outfile, test_outfile)
        self.dataset.transform_queries(lambda x: x[:max_query_len])
        self.dataset.transform_docs(lambda x: x[:max_doc_len])

    def _define_trainset(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('query', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('pos_doc', shape=(n_out_examples,), dtype=vlen_int64)

        neg_docs = dataset_fp.create_group('neg_docs')
        for i in range(self.dataset.trainset.num_neg_examples):
            neg_docs.create_dataset(str(i), shape=(n_out_examples,), dtype=vlen_int64)

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

        for i, doc in enumerate(neg_docs):
            fp['neg_docs'][str(i)][idx] = doc

    def _save_candidate_row(self, fp, q_id, query, doc, label, idx):
        fp['q_id'][idx] = q_id
        fp['query'][idx] = query
        fp['doc'][idx] = doc
        fp['label'][idx] = label


def main():
    class Args:
        # qa_utils dataset
        FIQA_DIR = 'D:\\datasets\\qa\\fiqa\\'
        SPLIT_FILE = 'D:\\datasets\\qa\\fiqa\\fiqa_split.pkl'
        num_neg_examples = 50

    fiqa = FiQA(Args())
    base = 'data/fiqa_neg_50'
    train = 'train.hdf5'
    dev = 'dev.hdf5'
    test = 'test.hdf5'
    tokenizer = NLTKTokenizer()
    saver = MANHdf5Saver(fiqa,
                         tokenizer,
                         22413,
                         os.path.join(base, 'vocabulary.pkl'),
                         20,
                         150,
                         os.path.join(base, train),
                         os.path.join(base, test),
                         os.path.join(base, dev))

    saver.build_all()


if __name__ == '__main__':
    main()
