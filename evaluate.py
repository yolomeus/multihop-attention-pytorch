import argparse

import torch
from torch.utils.data import DataLoader

from data_source import MultihopTestset
from model import QAMatching
from qa_utils.evaluation import read_args, evaluate_all
from qa_utils.io import get_cuda_device

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('DEV_DATA', help='Dev data hdf5 filepath.')
    ap.add_argument('TEST_DATA', help='Test data hdf5 filepath.')
    ap.add_argument('WORKING_DIR', help='Working directory containing args.csv and a ckpt folder.')
    ap.add_argument('--mrr_k', type=int, default=10, help='Compute MRR@k')
    ap.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    ap.add_argument('--interval', type=int, default=1, help='Only evaluate every i-th checkpoint.')
    ap.add_argument('--num_workers', type=int, default=1, help='number of workers used by the dataloader.')
    args = ap.parse_args()

    train_args = read_args(args.WORKING_DIR)

    dev_set = MultihopTestset(args.DEV_DATA)
    test_set = MultihopTestset(args.TEST_DATA)

    dev_dl = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                        num_workers=args.num_workers, collate_fn=dev_set.collate)
    test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                         num_workers=args.num_workers, collate_fn=test_set.collate)
    device = get_cuda_device()

    model = QAMatching(int(train_args['vocab_size']), int(train_args['embed_dim']), int(train_args['hidden_dim']))
    model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate_all(model, args.WORKING_DIR, dev_dl, test_dl, args.mrr_k, device, has_multiple_inputs=True,
                 interval=args.interval)
