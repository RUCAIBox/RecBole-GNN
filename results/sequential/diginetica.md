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
neg_sampling: ~
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
| **SR-GNN**           | 0.3881    | 0.1754 | 0.2253  | 0.3881 | 0.0388       |
| **GC-SAN**           | 0.4127    | 0.1881 | 0.2408  | 0.4127 | 0.0413       |
| **NISER+**           |           |        |         |        |              |
| **LESSR**            |           |        |         |        |              |
| **TAGNN**            |           |        |         |        |              |
| **GCE-GNN**          |           |        |         |        |              |

# Hyper-parameters

|                      | Best hyper-parameters                                                     | Tuning range                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **SR-GNN**            | learning_rate=0.001<br />step=1                              | learning_rate in [0.01, 0.001, 0.0001]<br />step in [1, 2]    |
| **GC-SAN**            | learning_rate=0.001<br />step=1                              | learning_rate in [0.01, 0.001, 0.0001]<br />step in [1, 2]    |
