# Experimental Setting

**Dataset:** [LastFM](http://files.grouplens.org/datasets/hetrec2011/)

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
| lastfm     | 1,893  | 12,524 | 186,479       | 99.21%   |

# Evaluation Results

| Method               | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
| -------------------- | --------- | ------ | ------- | ------ | ------------ |
| **DiffNet**          | 0.5784    | 0.6263 | 0.5943  | 0.7714 | 0.2499       |
| **GraphRec**         | 0.1982    | 0.2216 | 0.1768  | 0.4038 | 0.0828       |
| **SEPT**             | 0.5638    | 0.6991 | 0.6383  | 0.7659 | 0.2612       |

# Hyper-parameters

|                      | Best hyper-parameters                                                     | Tuning range                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DiffNet**           | learning_rate=0.005<br />n_layers=2                           | learning_rate in [0.01, 0.005, 0.001, 0.0005, 0.0001]<br />n_layers in [1, 2, 3]   |
| **GraphRec**          | learning_rate=0.005<br />mlp_layer_num=2                              | learning_rate in [0.005, 0.001, 0.0005]<br />mlp_layer_num in [1, 2, 3]    |
| **SEPT**             | learning_rate=0.005<br />n_layers=2<br />ssl_weight=1e-05                              | learning_rate in [0.005, 0.001, 0.0005]<br />n_layers in [1, 2, 3]<br />ssl_weight in [1e-2, 5e-3, 1e-3, 1e-4, 1e-5]    |
