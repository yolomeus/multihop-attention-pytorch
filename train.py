from argparse import ArgumentParser

import torch
from torch import optim
from torch.utils.data import DataLoader

from data_source import MultihopTrainset
from model import QAMatching
from qa_utils.io import get_cuda_device, load_pkl_file
from qa_utils.training import train_model_pairwise


def max_margin(pos_sim, neg_sim, margin=0.2):
    return torch.clamp(margin - pos_sim + neg_sim, min=0)


def main():
    ap = ArgumentParser(description='Train the DUET model.')
    ap.add_argument('TRAIN_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('VOCAB_FILE', help='Pickle file containing the mapping from ids to words.')

    ap.add_argument('--hidden_dim', type=int, default=512,
                    help='The hidden dimension used throughout the whole network.')

    ap.add_argument('--embed_dim', type=int, default=300, help='The dimensionality of the GloVe embeddings.')
    ap.add_argument('--glove_cache', default='glove_cache', help='Glove cache directory.')

    ap.add_argument('--num_neg_examples', type=int, default=50,
                    help='Number of documents to sample document with maximum loss from.')
    ap.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')

    ap.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    ap.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    ap.add_argument('--pred_batch_size', type=int, default=None, help='Batch size when doing forward pass to determine '
                                                                      'negative examples with maximum loss.')

    ap.add_argument('--accumulate_batches', type=int, default=1,
                    help='Update weights after this many batches')
    ap.add_argument('--working_dir', default='train', help='Working directory for checkpoints and logs.')
    ap.add_argument('--random_seed', type=int, default=1579129142, help='Random seed.')
    ap.add_argument('--num_workers', type=int, default=1, help='Number of workers used for loading data.')

    args = ap.parse_args()

    torch.manual_seed(args.random_seed)

    trainset = MultihopTrainset(args.TRAIN_DATA, args.num_neg_examples)
    train_dl = DataLoader(trainset, args.batch_size, True, collate_fn=trainset.collate, num_workers=args.num_workers)

    device = get_cuda_device()

    id_to_word = load_pkl_file(args.VOCAB_FILE)
    vocab_size = len(id_to_word.keys())
    model = QAMatching(vocab_size, args.embed_dim, args.hidden_dim, id_to_word, args.glove_cache)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model_pairwise(model, max_margin, train_dl, optimizer, args, device, args.num_neg_examples,
                         has_multiple_inputs=True)


if __name__ == '__main__':
    main()
