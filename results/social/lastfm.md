# Experimental Setting

**Dataset:** [LastFM](http://files.grouplens.org/datasets/hetrec2011/)

> Note that datasets for social recommendation methods can be downloaded from [Social-Datasets](https://github.com/Sherry-XLL/Social-Datasets).

**Filtering:** None

**Evaluation:** ratio-based 8:1:1, full sort

**Metrics:** Recall@10, NGCG@10, MRR@10, Hit@10, Precision@10

**Properties:**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: artist_id
NET_SOURCE_ID_FIELD: source_id
NET_TARGET_ID_FIELD: target_id
LABEL_FIELD: label
NEG_PREFIX: neg_
load_col:
  inter: [user_id, artist_id]
  net: [source_id, target_id]

# social network config
filter_net_by_inter: True
undirected_net: True

# training and evaluation
epochs: 5000
train_batch_size: 4096
eval_batch_size: 409600000
valid_metric: NDCG@10
stopping_step: 50
```

For fairness, we restrict users' and items' embedding dimension as following. Please adjust the name of the corresponding args of different models.
```
embedding_size: 64
```

# Dataset Statistics

| Dataset    | #Users | #Items | #Interactions | Sparsity |
| ---------- | ------ | ------ | ------------- | -------- |
| lastfm     | 1,892  | 17,632 | 92,834        | 99.72%   |

# Evaluation Results

| Method               | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
| -------------------- | --------- | ------ | ------- | ------ | ------------ |
| **BPR**              | 0.1761    | 0.3026 | 0.1674  | 0.5573 | 0.0858       |
| **NeuMF**            | 0.1696    | 0.2924 | 0.1604  | 0.5456 | 0.0828       |
| **NGCF**             | 0.1960    | 0.3479 | 0.1898  | 0.6141 | 0.0961       |
| **LightGCN**         | 0.2064    | 0.3559 | 0.1972  | 0.6322 | 0.1009       |
| **DiffNet**          | 0.1757    | 0.3117 | 0.1694  | 0.5621 | 0.0857       |
| **MHCN**             | 0.2123    | 0.3782 | 0.2068  | 0.6523 | 0.1042       |
| **SEPT**             | 0.2127    | 0.3703 | 0.2057  | 0.6465 | 0.1044       |

# Hyper-parameters

|                      | Best hyper-parameters                                                     | Tuning range                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **BPR**             | learning_rate=0.0005                              | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]    |
| **NeuMF**           | learning_rate=0.0005<br />dropout_prob=0.1                           | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />dropout_prob in [0.1, 0.2, 0.3]   |
| **NGCF**              | learning_rate=0.0005<br />hidden_size_list=[64,64,64]                              | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />hidden_size_list in ['[64]', '[64,64]', '[64,64,64]']    |
| **LightGCN**             | learning_rate=0.001<br />n_layers=3                              | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />n_layers in [1, 2, 3]    |
| **DiffNet**           | learning_rate=0.0005<br />n_layers=1                           | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />n_layers in [1, 2, 3]   |
| **MHCN**              | learning_rate=0.0005<br />n_layers=2<br />ssl_reg=1e-05                              | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />n_layers in [1, 2, 3]<br />ssl_reg in [1e-04, 1e-05, 1e-06]    |
| **SEPT**             | learning_rate=0.0005<br />n_layers=2<br />ssl_weight=1e-07                              | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />n_layers in [1, 2, 3]<br />ssl_weight in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]    |
