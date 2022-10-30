# Experimental Setting

**Dataset:** diginetica-not-merged

**Filtering:** Remove users and items with less than 5 interactions

**Evaluation:** leave one out, full sort

**Metrics:** Recall@10, NGCG@10, MRR@10, Hit@10, Precision@10

**Properties:**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 20
POSITION_FIELD: position_id
load_col:
  inter: [session_id, item_id, timestamp]
user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 2000
valid_metric: MRR@10
eval_args:
  split: {'LS':"valid_and_test"}
  mode: full
  order: TO
train_neg_sample_args: ~
```

For fairness, we restrict users' and items' embedding dimension as following. Please adjust the name of the corresponding args of different models.
```
embedding_size: 64
```

# Dataset Statistics

| Dataset    | #Users | #Items | #Interactions | Sparsity |
| ---------- | ------ | ------ | ------------- | -------- |
| diginetica | 72,014 | 29,454 | 580,490       | 99.97%   |

# Evaluation Results

| Method               | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
| -------------------- | --------- | ------ | ------- | ------ | ------------ |
| **GRU4Rec**          | 0.3691    | 0.1632 | 0.2114  | 0.3691 | 0.0369       |
| **NARM**             | 0.3801    | 0.1695 | 0.2188  | 0.3801 | 0.0380       |
| **SASRec**           | 0.4144    | 0.1857 | 0.2393  | 0.4144 | 0.0414       |
| **SR-GNN**           | 0.3881    | 0.1754 | 0.2253  | 0.3881 | 0.0388       |
| **GC-SAN**           | 0.4127    | 0.1881 | 0.2408  | 0.4127 | 0.0413       |
| **NISER+**           | 0.4144    | 0.1904 | 0.2430  | 0.4144 | 0.0414       |
| **LESSR**            | 0.3964    | 0.1763 | 0.2279  | 0.3964 | 0.0396       |
| **TAGNN**            | 0.3894    | 0.1763 | 0.2263  | 0.3894 | 0.0389       |
| **GCE-GNN**          | 0.4284    | 0.1961 | 0.2507  | 0.4284 | 0.0428       |
| **SGNN-HN**          | 0.4183    | 0.1877 | 0.2418  | 0.4183 | 0.0418       |

# Hyper-parameters

|                      | Best hyper-parameters                                                     | Tuning range                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **GRU4Rec** | learning_rate=0.01<br />hidden_size=128<br />dropout_prob=0.3<br />num_layers=1 | learning_rate in [1e-2, 1e-3, 3e-3]<br />num_layers in [1, 2, 3]<br />hidden_size in [128]<br />dropout_prob in [0.1, 0.2, 0.3] |
| **SASRec**           | learning_rate=0.001<br />n_layers=2<br />attn_dropout_prob=0.2<br />hidden_dropout_prob=0.2 | learning_rate in [0.001, 0.0001]<br />n_layers in [1, 2]<br />hidden_dropout_prob in [0.2, 0.5]<br />attn_dropout_prob in [0.2, 0.5] |
| **NARM**             | learning_rate=0.001<br />hidden_size=128<br />n_layers=1<br />dropout_probs=[0.25, 0.5] | learning_rate in [0.001, 0.01, 0.03]<br />hidden_size in [128]<br />n_layers in [1, 2]<br />dropout_probs in ['[0.25,0.5]', '[0.2,0.2]', '[0.1,0.2]'] |
| **SR-GNN**            | learning_rate=0.001<br />step=1                              | learning_rate in [0.01, 0.001, 0.0001]<br />step in [1, 2]    |
| **GC-SAN**            | learning_rate=0.001<br />step=1                              | learning_rate in [0.01, 0.001, 0.0001]<br />step in [1, 2]    |
| **NISER+**            | learning_rate=0.001<br />sigma=16                              | learning_rate in [0.01, 0.001, 0.003]<br />sigma in [10, 16, 20]    |
| **LESSR**            | learning_rate=0.001<br />n_layers=4                              | learning_rate in [0.01, 0.001, 0.003]<br />n_layers in [2, 4]    |
| **TAGNN**            | learning_rate=0.001                              | learning_rate in [0.01, 0.001, 0.003]<br />train_batch_size=512    |
| **GCE-GNN**            | learning_rate=0.001<br />dropout_global=0.5                              | learning_rate in [0.01, 0.001, 0.003]<br />dropout_global in [0.2, 0.5]    |
| **SGNN-HN**            | learning_rate=0.003<br />scale=12<br />step=2                              | learning_rate in [0.01, 0.001, 0.003]<br />scale in [12, 16, 20]<br />step in [2, 4, 6]    |
