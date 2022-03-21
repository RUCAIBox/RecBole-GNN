# RecBole-Graph

**RecBole-Graph** is a library built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing recommendation algorithms based on graph neural networks (GNNs). Our library includes algorithms covering three major categories:
* **General Recommendation** with user-item interaction graphs;
* **Sequential Recommendation** with session/sequence graphs;
* **Social Recommendation** with social networks.

## Highlights

* **Easy-to-use and unified API**:
    Our library shares unified API and input (atomic files) as RecBole.
* **Efficient and reusable graph processing**:
    We provide highly efficient and reusable basic datasets, dataloaders and layers for graph processing and learning.
* **Extensive graph library**:
    Graph neural networks from widely-used library like [PyG](https://github.com/pyg-team/pytorch_geometric) or [DGL](https://github.com/dmlc/dgl) are incorporated. Recently proposed graph algorithms can be easily equipped and compared with existing methods.

## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole_graph.py
```

If you want to change the models or datasets, just run the script by setting additional command parameters:

```bash
python run_recbole_graph.py -m [model] -d [dataset]
```

## Implemented Models

We list currently supported models according to category:

**General Recommendation**:

* **[NGCF](recbole_graph/model/general_recommender/ngcf.py)** from Wang *et al.*: [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108) (SIGIR 2019).
* **[LightGCN](recbole_graph/model/general_recommender/lightgcn.py)** from He *et al.*: [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) (SIGIR 2020).
* **[SGL](recbole_graph/model/general_recommender/sgl.py)** from Wu *et al.*: [Self-supervised Graph Learning for Recommendation](https://arxiv.org/abs/2010.10783) (SIGIR 2021).
* **[HMLET](recbole_graph/model/general_recommender/hmlet.py)** from Kong *et al.*: [Linear, or Non-Linear, That is the Question!](https://arxiv.org/abs/2111.07265) (WSDM 2022).
* **[NCL](recbole_graph/model/general_recommender/ncl.py)** from Lin *et al.*: [Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning](https://arxiv.org/abs/2202.06200) (TheWebConf 2022).

**Sequential Recommendation**:

* **[SR-GNN](recbole_graph/model/sequential_recommender/srgnn.py)** from Wu *et al.*: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855) (AAAI 2019).
* **[GC-SAN](recbole_graph/model/sequential_recommender/gcsan.py)** from Xu *et al.*: [Graph Contextualized Self-Attention Network for Session-based Recommendation](https://www.ijcai.org/proceedings/2019/547) (IJCAI 2019).
* **[NISER+](recbole_graph/model/sequential_recommender/niser.py)** from Gupta *et al.*: [NISER: Normalized Item and Session Representations to Handle Popularity Bias](https://arxiv.org/abs/1909.04276) (GRLA, CIKM 2019 workshop).
* **[LESSR](recbole_graph/model/sequential_recommender/lessr.py)** from Chen *et al.*: [Handling Information Loss of Graph Neural Networks for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3394486.3403170) (KDD 2020).
* **[TAGNN](recbole_graph/model/sequential_recommender/tagnn.py)** from Yu *et al.*: [TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation](https://arxiv.org/abs/2005.02844) (SIGIR 2020 short).

**Social Recommendation**:

* **[DiffNet](recbole_graph/model/social_recommender/diffnet.py)** from Wu *et al.*: [A Neural Influence Diffusion Model for Social Recommendation](https://arxiv.org/abs/1904.10322) (SIGIR 2019).

## The Team

RecBole-Graph is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers are Yupeng Hou ([@hyp1231](https://github.com/hyp1231)), Lanling Xu ([@Sherry-XLL](https://github.com/Sherry-XLL)) and Changxin Tian ([@ChangxinTian](https://github.com/ChangxinTian)).
