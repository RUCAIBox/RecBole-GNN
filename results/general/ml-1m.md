# Experimental Setting

**Dataset:** [MovieLens-1M](https://grouplens.org/datasets/movielens/)

**Filtering:** Remove interactions with a rating score of less than 3

**Evaluation:** ratio-based 8:1:1, full sort

**Metrics:** Recall@10, NGCG@10, MRR@10, Hit@10, Precision@10

**Properties:**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
val_interval:
    rating: "[3,inf)"
unused_col: 
    inter: [rating]

# training and evaluation
epochs: 500
train_batch_size: 4096
valid_metric: MRR@10
eval_batch_size: 4096000
```

For fairness, we restrict users' and items' embedding dimension as following. Please adjust the name of the corresponding args of different models.
```
embedding_size: 64
```

# Dataset Statistics

| Dataset    | #Users | #Items | #Interactions | Sparsity |
| ---------- | ------ | ------ | ------------- | -------- |
| ml-1m      | 6,040  | 3,629  | 836,478       | 96.18%   |

# Evaluation Results

| Method       | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
|--------------|-----------|--------|---------|--------|--------------|
| **BPR**      | 0.1776    | 0.4187 | 0.2401  | 0.7199 | 0.1779       |
| **NeuMF**    | 0.1651    | 0.4020 | 0.2271  | 0.7029 | 0.1700       |
| **NGCF**     | 0.1814    | 0.4354 | 0.2508  | 0.7239 | 0.1850       |
| **LightGCN** | 0.1861    | 0.4388 | 0.2538  | 0.7330 | 0.1863       |
| **LightGCL** | 0.1867    | 0.4283 | 0.2479  | 0.7370 | 0.1815       |
| **SGL**      | 0.1889    | 0.4315 | 0.2505  | 0.7392 | 0.1843       |
| **HMLET**    | 0.1847    | 0.4297 | 0.2490  | 0.7305 | 0.1836       |
| **NCL**      | 0.2021    | 0.4599 | 0.2702  | 0.7565 | 0.1962       |
| **SimGCL**   | 0.2029    | 0.4550 | 0.2667  | 0.7640 | 0.1933       |
| **XSimGCL**  | 0.2116    | 0.4638 | 0.2750  | 0.7743 | 0.1987       |

# Hyper-parameters

|              | Best hyper-parameters                                                                                                              | Tuning range                                                                                                                                                                                                                   |
|--------------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **BPR**      | learning_rate=0.001                                                                                                                | learning_rate choice [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]                                                                                                                |
| **NeuMF**    | learning_rate=0.0001<br />mlp_hidden_size=[32,16,8]<br />dropout_prob=0                                                            | learning_rate choice [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]<br/>mlp_hidden_size choice ['[64,64]', '[64,32]', '[64,32,16]','[32,16,8]']<br/>dropout_prob choice [0, 0.1, 0.2]                                  |
| **NGCF**     | learning_rate=0.0002<br />message_dropout=0.0<br />node_dropout=0.0                                                                | learning_rate choice [0.001, 0.0005, 0.0002]<br/>node_dropout choice [0.0, 0.1]<br/>message_dropout choice [0.0, 0.1]                                                                                                          |
| **LightGCN** | learning_rate=0.002<br />n_layers=3<br />reg_weight=0.0001                                                                         | learning_rate choice [0.005, 0.002, 0.001]<br/>n_layers choice [2, 3]<br/>reg_weight choice [1e-4, 1e-5]                                                                                                                       |
| **LightGCL** | learning_rate=0.001<br />n_layers=2<br />lambda1=0.0001<br />temp=2<br />lambda2=1e-7<br />dropout=0.1                             | learning_rate choice [0.001]<br/>n_layers choice [2, 3]<br/>lambda1 choice [0.01, 0.005, 0.001, 0.0001, 1e-5, 1e-7]<br/>temp choice [0.5, 0.8, 2, 3]<br/>lambda2 choice [1e-4, 1e-5, 1e-7]<br/>dropout choice [0.0, 0.1, 0.25] |
| **SGL**      | learning_rate=0.002<br />n_layers=3<br />reg_weight=0.0001<br />ssl_tau=0.5<br />drop_ratio=0.1<br />ssl_weight=0.005              | learning_rate choice [0.002]<br/>n_layers choice [3]<br/>reg_weight choice [1e-4]<br/>ssl_tau choice [0.1, 0.5]<br/>drop_ratio choice [0.1, 0.3]<br/>ssl_weight choice [1e-5, 1e-6, 1e-7, 0.005, 0.01, 0.05]                   |
| **HMLET**    | learning_rate=0.002<br />n_layers=4<br />activation_function=leakyrelu                                                             | learning_rate choice [0.002, 0.001, 0.0005]<br/>n_layers choice [3, 4]<br/>activation_function choice ['elu', 'leakyrelu']                                                                                                     |
| **NCL**      | learning_rate=0.002<br />n_layers=3<br />reg_weight=0.0001<br />ssl_temp=0.1<br />ssl_reg=1e-06<br />hyper_layers=1<br />alpha=1.5 | learning_rate choice [0.002]<br/>n_layers choice [3]<br/>reg_weight choice [1e-4]<br/>ssl_temp choice [0.1, 0.05]<br/>ssl_reg choice [1e-7, 1e-6]<br/>hyper_layers choice [1]<br/>alpha choice [1, 0.8, 1.5]                   |
| **SimGCL**   | learning_rate=0.002<br />n_layers=2<br />reg_weight=0.0001<br />temperature=0.05<br />lambda=1e-5<br />eps=0.1                     | learning_rate choice [0.002]<br/>n_layers choice [2, 3]<br/>reg_weight choice [1e-4]<br/>temperature choice [0.05, 0.1, 0.2]<br/>lambda choice [1e-5, 1e-6, 1e-7, 0.005, 0.01, 0.05]<br/>eps choice [0.1, 0.2]                 |
| **XSimGCL** | learning_rate=0.002<br />n_layers=2<br />reg_weight=0.0001<br />temperature=0.2<br />lambda=0.1<br />eps=0.2<br />layer_cl=1 | learning_rate choice [0.002]<br/>n_layers choice [2, 3]<br/>reg_weight choice [1e-4]<br/>temperature choice [0.05, 0.1, 0.2]<br/>lambda choice [1e-5, 1e-6, 1e-7, 1e-4, 0.005, 0.01, 0.05, 0.1]<br/>eps choice [0.1, 0.2]<br/>layer_cl choice [1] |
