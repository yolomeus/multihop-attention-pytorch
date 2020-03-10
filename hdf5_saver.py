import h5py
import numpy as np

from qa_utils.preprocessing.hdf5saver import Hdf5Saver


class MANHdf5Saver(Hdf5Saver):
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
