from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model import MultiHopAttentionRanker
from qa_utils.io import load_json_file


def max_margin(pos_sim, neg_sim, margin=0.2):
    return torch.clamp(margin - pos_sim + neg_sim, min=0)


def main():
    ap = ArgumentParser(description='Train the DUET model.')
    ap.add_argument('TRAIN_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('VAL_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('TEST_DATA', help='Path to an hdf5 file containing the training data.')

    ap.add_argument('--accumulate_grad_batches', type=int, default=1,
                    help='Update weights after this many batches')
    ap.add_argument('--gpus', type=int, nargs='+', help='GPU IDs to train on')
    ap.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    ap.add_argument('--logdir', default='logs', help='Directory for checkpoints and logs')

    ap.add_argument('--val_patience', type=int, default=3, help='Validation patience')
    ap.add_argument('--random_seed', type=int, default=1579129142, help='Random seed.')

    ap = MultiHopAttentionRanker.add_model_specific_args(ap)
    args = ap.parse_args()

    seed_everything(args.random_seed)

    id_to_word = load_json_file(args.VOCAB_FILE)
    vocab_size = len(id_to_word.keys())
    model = MultiHopAttentionRanker(vocab_size,
                                    args.embed_dim,
                                    args.hidden_dim,
                                    args.learning_rate,
                                    args.loss_margin,
                                    id_to_word,
                                    args.glove_cache,
                                    args.TRAIN_DATA,
                                    args.VAL_DATA,
                                    args.TEST_DATA,
                                    args.batch_size,
                                    args.num_neg_examples)

    early_stopping = EarlyStopping('val_mrr', mode='max', patience=args.val_patience)
    model_checkpoint = ModelCheckpoint(monitor='val_mrr', mode='max')
    # DDP seems to be buggy currently, so we use DP for now
    trainer = Trainer.from_argparse_args(args, distributed_backend='dp',
                                         default_root_dir=args.logdir,
                                         early_stop_callback=early_stopping,
                                         checkpoint_callback=model_checkpoint)
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
