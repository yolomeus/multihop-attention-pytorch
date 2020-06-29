import os
import random
from argparse import ArgumentParser

import numpy as np
import h5py

from qa_utils.preprocessing.antique import Antique
from qa_utils.preprocessing.base_hdf5_saver import BaseHdf5Saver
from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.insrqa import InsuranceQA
from qa_utils.preprocessing.msmarco import MSMARCO
from tokenizer import NLTKTokenizer


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


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('DATA_DIR', type=str, help='Directory with the raw dataset files.')
    ap.add_argument('OUTPUT_DIR', type=str, help='Directory to store the generated files in.')
    ap.add_argument('DATA_SET', type=str, choices=['FIQA', 'MSMARCO', 'ANTIQUE', 'INSURANCE_QA'],
                    help='The dataset that will be processed.')
    ap.add_argument('--random_seed', type=int, default=1586530105, help='the global random seed')
    ap.add_argument('--vocab_size', type=int, default=80000,
                    help='Only use the n most frequent words for the vocabulary.')
    ap.add_argument('--num_neg_examples', type=int, default=50,
                    help='For each query sample this many negative documents.')
    ap.add_argument('--max_q_len', type=int, default=20, help='Maximum query length.')
    ap.add_argument('--max_d_len', type=int, default=150, help='Maximum document legth.')
    ap.add_argument('--examples_per_query', type=int, choices=[100, 500, 1000, 1500],
                    default=500, help='How many examples per query in the dev- and testset for insurance qa.')

    ap.add_argument('--no_train', default=False, action='store_true', help='Don\'t export the train set.')
    ap.add_argument('--no_dev', default=False, action='store_true', help='Don\'t export the dev set.')
    ap.add_argument('--no_test', default=False, action='store_true', help='Don\'t export the test set.')

    args = ap.parse_args()

    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    random.seed(args.random_seed)

    split_dir = 'qa_utils/splits'
    if args.DATA_SET == 'FIQA':
        args.FIQA_DIR = args.DATA_DIR
        args.SPLIT_FILE = os.path.join(split_dir, 'fiqa_split.pkl')
        dataset = FiQA(args)
    elif args.DATA_SET == 'MSMARCO':
        args.MSM_DIR = args.DATA_DIR
        args.MSM_SPLIT = os.path.join(split_dir, 'msm_split.pkl')
        dataset = MSMARCO(args)
    elif args.DATA_SET == 'INSURANCE_QA':
        args.INSRQA_V2_DIR = args.DATA_DIR
        dataset = InsuranceQA(args)
    elif args.DATA_SET == 'ANTIQUE':
        args.ANTIQUE_DIR = args.DATA_DIR
        args.SPLIT_FILE = os.path.join(split_dir, 'antique_split.pkl')
        dataset = Antique(args)

    else:
        raise NotImplementedError()

    train_path = None if args.no_train else os.path.join(args.OUTPUT_DIR, 'train.hdf5')
    dev_path = None if args.no_dev else os.path.join(args.OUTPUT_DIR, 'dev.hdf5')
    test_path = None if args.no_test else os.path.join(args.OUTPUT_DIR, 'test.hdf5')

    saver = MANHdf5Saver(dataset,
                         NLTKTokenizer(),
                         args.vocab_size,
                         os.path.join(args.OUTPUT_DIR, 'vocabulary.json'),
                         train_outfile=train_path,
                         dev_outfile=dev_path,
                         test_outfile=test_path,
                         max_doc_len=args.max_d_len,
                         max_query_len=args.max_q_len,
                         n_neg_examples=args.num_neg_examples)

    saver.build_all()
