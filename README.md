# Multihop Attention Network for Question Answer Matching
This repository contains an implementation for training the Multi-Hop Attention Network proposed in 
[Multihop Attention Networks for Question Answer Matching](https://dl.acm.org/doi/10.1145/3209978.3210009). The model's 
implementation is mostly based on code from [this respository](https://github.com/namkhanhtran/nn4nqa/). 


The following datasets are currently supported:
* [MS MARCO Ranking](http://www.msmarco.org/dataset.aspx)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
* [WikiPassageQA](https://sites.google.com/site/lyangwww/code-data)

## Usage
First preprocess the desired dataset using generate_hdf5.py:
```
python generate_hdf5.py [-h] [--vocab_size VOCAB_SIZE]
                        [--num_neg_examples NUM_NEG_EXAMPLES]
                        [--max_q_len MAX_Q_LEN] [--max_d_len MAX_D_LEN]
                        [--examples_per_query {100,500,1000,1500}]
                        [--no_train] [--no_dev] [--no_test]
                        DATA_DIR OUTPUT_DIR
                        {FIQA,MSMARCO,ANTIQUE,INSURANCE_QA}
```
e.g.:
```
python /path/to/raw/insrqa /output/path/insrqa INSURANCE_QA --num_neg_examples 50 --vocab_size 75000
```
Explanation of the arguments:
```
positional arguments:
  DATA_DIR              Directory with the raw dataset files.
  OUTPUT_DIR            Directory to store the generated files in.
  {FIQA,MSMARCO,ANTIQUE,INSURANCE_QA}
                        The dataset that will be processed.

optional arguments:
  -h, --help            show this help message and exit
  --vocab_size VOCAB_SIZE
                        Only use the n most frequent words for the vocabulary.
  --num_neg_examples NUM_NEG_EXAMPLES
                        For each query sample this many negative documents.
  --max_q_len MAX_Q_LEN
                        Maximum query length.
  --max_d_len MAX_D_LEN
                        Maximum document legth.
  --examples_per_query {100,500,1000,1500}
                        How many examples per query in the dev- and testset
                        for insurance qa.
  --no_train            Don't export the train set.
  --no_dev              Don't export the dev set.
  --no_test             Don't export the test set.
```
#
### Training
You can then point the `train.py` script to the generated data and start training: 
```
train.py [-h] [--hidden_dim HIDDEN_DIM] [--embed_dim EMBED_DIM]
                [--glove_cache GLOVE_CACHE]
                [--num_neg_examples NUM_NEG_EXAMPLES]
                [--learning_rate LEARNING_RATE] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--pred_batch_size PRED_BATCH_SIZE]
                [--accumulate_batches ACCUMULATE_BATCHES]
                [--working_dir WORKING_DIR] [--random_seed RANDOM_SEED]
                [--num_workers NUM_WORKERS]
                TRAIN_DATA VOCAB_FILE
```
e.g.:
```
python /output/path/insrqa/train.hdf5 output/path/insrqa/vocabulary.pkl --epochs 3 --batch_size 256 --working_dir ./train_logs
```
Arguments:
```
positional arguments:
  TRAIN_DATA            Path to an hdf5 file containing the training data.
  VOCAB_FILE            Pickle file containing the mapping from ids to words.

optional arguments:
  -h, --help            show this help message and exit
  --hidden_dim HIDDEN_DIM
                        The hidden dimension used throughout the whole
                        network.
  --embed_dim EMBED_DIM
                        The dimensionality of the GloVe embeddings.
  --glove_cache GLOVE_CACHE
                        Glove cache directory.
  --num_neg_examples NUM_NEG_EXAMPLES
                        Number of documents to sample document with maximum
                        loss from.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --epochs EPOCHS       Number of epochs.
  --batch_size BATCH_SIZE
                        Batch size.
  --pred_batch_size PRED_BATCH_SIZE
                        Batch size when doing forward pass to
                        determine negative examples with maximum loss.
  --accumulate_batches ACCUMULATE_BATCHES
                        Update weights after this many batches
  --working_dir WORKING_DIR
                        Working directory for checkpoints and logs.
  --random_seed RANDOM_SEED
                        Random seed.
  --num_workers NUM_WORKERS
                        Number of workers used for loading data.
```
#
### Evaluation
Use the `evaluate.py` script to evaluate all checkpoints that were saved to the training directory:
```
evaluate.py [-h] [--mrr_k MRR_K] [--batch_size BATCH_SIZE]
                   [--interval INTERVAL] [--num_workers NUM_WORKERS]
                   DEV_DATA TEST_DATA WORKING_DIR
```

e.g.:
```
/output/path/insrqa/dev.hdf5 /output/path/insrqa/test.hdf5 ./train_logs --batch_size 2048
```
Arguments:
```
positional arguments:
  DEV_DATA              Dev data hdf5 filepath.
  TEST_DATA             Test data hdf5 filepath.
  WORKING_DIR           Working directory containing args.csv and a ckpt
                        folder.

optional arguments:
  -h, --help            show this help message and exit
  --mrr_k MRR_K         Compute MRR@k
  --batch_size BATCH_SIZE
                        Batch size
  --interval INTERVAL   Only evaluate every i-th checkpoint.
  --num_workers NUM_WORKERS
                        number of workers used by the dataloader.
```