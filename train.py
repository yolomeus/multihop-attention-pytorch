import csv
import os
from argparse import ArgumentParser

import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_source import MultihopTrainset
from model import QAMatching
from qa_utils.io import get_cuda_device, list_to, load_pkl_file
from qa_utils.misc import Logger


def max_margin(pos_sim, neg_sim, margin=0.2):
    return torch.clamp(margin - pos_sim + neg_sim, min=0)


def sample_neg_docs(model, device, query_batch, query_lens, pos_doc_batch, pos_lens, neg_doc_batches, neg_lens_batches):
    """Compute losses for each sampled negative document and return batch of negative documents with maximum loss.

    Returns: tuple(list, list): a batch of (padded) negative documents and a list of their non padded lens.
    """
    # we compute losses for all sampled negative documents without storing intermediate activations
    with torch.no_grad():
        pos_sim, neg_sims = model(query_batch.to(device),
                                  query_lens.to(device),
                                  pos_doc_batch.to(device),
                                  pos_lens.to(device),
                                  list_to(device, neg_doc_batches),
                                  list_to(device, neg_lens_batches))

        losses = [max_margin(pos_sim, neg_sim) for neg_sim in neg_sims]
        losses = torch.stack(losses, dim=1)
        # index of the highest loss negative document for each row in the batch
        max_doc_ids = torch.argmax(losses, dim=1)

        # build a negative doc batch with highest loss docs
        max_doc_batch = []
        max_doc_lens = []
        for j, idx in enumerate(max_doc_ids):
            max_doc = neg_doc_batches[idx][j]
            max_doc_len = neg_lens_batches[idx][j]

            max_doc_batch.append(max_doc)
            max_doc_lens.append(max_doc_len)
        max_doc_batch = pad_sequence(max_doc_batch, batch_first=True, padding_value=0)
        max_doc_lens = torch.stack(max_doc_lens)

        return max_doc_batch, max_doc_lens


def train(model, train_dl, optimizer, device, args):
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    # save all args in a file
    args_file = os.path.join(args.working_dir, 'args.csv')
    print('writing {}...'.format(args_file))
    with open(args_file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            query_batch, query_lens, pos_doc_batch, pos_lens, neg_doc_batches, neg_lens_batches = batch

            # we only take negative documents with maximum loss
            neg_doc_batch, neg_doc_lens = sample_neg_docs(model, device, query_batch, query_lens, pos_doc_batch,
                                                          pos_lens, neg_doc_batches, neg_lens_batches)
            # the actual forward pass which stores activations
            pos_sim, neg_sims = model(query_batch.to(device),
                                      query_lens.to(device),
                                      pos_doc_batch.to(device),
                                      pos_lens.to(device),
                                      [neg_doc_batch],
                                      [neg_doc_lens])

            # we only passed a list of 1 negative docs for updating gradients
            neg_sim = neg_sims[0]
            batch_losses = max_margin(pos_sim, neg_sim)
            loss = torch.mean(batch_losses)
            loss = loss / args.accumulate_batches
            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        print('epoch {} -- loss: {}'.format(epoch + 1, epoch_loss))
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)


def main():
    ap = ArgumentParser(description='Train the DUET model.')
    ap.add_argument('TRAIN_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('VOCAB_FILE', help='Pickle file containing the mapping from ids to words.')

    ap.add_argument('--hidden_dim', type=int, default=512,
                    help='The hidden dimension used throughout the whole network.')

    ap.add_argument('--embed_dim', type=int, default=300, help='The dimensionality of the GloVe embeddings')
    ap.add_argument('--glove_cache', default='glove_cache', help='Glove cache directory.')

    ap.add_argument('--num_neg_examples', type=int, default=50,
                    help='Number of documents to sample document with maximum loss from')
    ap.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    ap.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    ap.add_argument('--batch_size', type=int, default=100, help='Batch size')
    ap.add_argument('--accumulate_batches', type=int, default=1,
                    help='Update weights after this many batches')
    ap.add_argument('--working_dir', default='train', help='Working directory for checkpoints and logs')
    ap.add_argument('--random_seed', type=int, default=1579129142, help='Random seed')

    args = ap.parse_args()

    torch.manual_seed(args.random_seed)

    trainset = MultihopTrainset(args.TRAIN_DATA, args.num_neg_examples)
    train_dl = DataLoader(trainset, args.batch_size, True, collate_fn=trainset.collate)

    device = get_cuda_device()

    id_to_word = load_pkl_file(args.VOCAB_FILE)
    vocab_size = len(id_to_word.keys())
    model = QAMatching(vocab_size, args.embed_dim, args.hidden_dim, id_to_word, args.glove_cache)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model, train_dl, optimizer, device, args)


if __name__ == '__main__':
    main()
