
![1301717130101_ pic](https://github.com/zyl24/OpenFGL/assets/59046279/e21b410f-2b5d-4515-8ab5-a176f98805a7)



**OpenFGL** (Open Federated Graph Learning) is a comprehensive, user-friendly algorithm library, complemented by an integrated evaluation platform, designed specifically for researchers in the field of federated graph learning (FGL).



[![Stars](https://img.shields.io/github/stars/zyl24/OpenFGL.svg?color=orange)](https://github.com/zyl24/OpenFGL/stargazers) ![](https://img.shields.io/github/last-commit/zyl24/OpenFGL) 
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992) -->

 



## Library Highlights :rocket: 

- 2 FGL Scenarios: Graph-FL and Subgraph-FL
- 10+ FGL Algorithms
- 34 FGL Datasets
- 12 GNN Models
- 5 Downstream Tasks
- Comprehensive FGL Data Property Analysis




## FGL Studies
Here we present a summary of papers in the FGL field.






<details>
  <summary>Graph-FL</summary>
    
| Title | Venue | Year | Materials |
| ----- | ----- | ---- | --------- |
| Federated Graph Classification over Non-IID Graphs | NeurIPS  | 2021 | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) [[Code]](https://github.com/Oxfordblue7/GCFL)  |
|Federated Learning on Non-IID Graphs via Structural Knowledge Sharing| AAAI| 2023| [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26187) [[Code]](https://github.com/yuetan031/fedstar) |

    
    
</details>


<details>
  <summary>Subgraph-FL</summary>
    
| Title | Venue | Year | Materials |
| ----- | ----- | ---- | --------- |
| Subgraph Federated Learning with Missing Neighbor Generation | NeurIPS  | 2021 | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/34adeb8e3242824038aa65460a47c29e-Abstract.html) [[Code]](https://github.com/zkhku/fedsage)    |
|FedGSL: Federated Graph Structure Learning for Local Subgraph Augmentation | ICBD| 2022| [[Paper]](https://ieeexplore.ieee.org/document/10020771) |
| Federated Node Classification over Graphs with Latent Link-type Heterogeneity| WWW|2023 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583471) [[Code]](https://github.com/Oxfordblue7/FedLIT)|
| FedHGN: a federated framework for heterogeneous graph neural networks| IJCAI| 2023 | [[Paper]](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/412) [[Code]](https://github.com/cynricfu/FedHGN)|
| Federated graph semantic and structural learning| IJCAI|2023 | [[Paper]](https://www.ijcai.org/proceedings/2023/0426.pdf) [[Code]](https://github.com/WenkeHuang/FGSSL)|
| Globally Consistent Federated Graph Autoencoder for Non-IID Graphs| IJCAI |2023 | [[Paper]](https://www.ijcai.org/proceedings/2023/0419.pdf) [[Code]](https://github.com/gcfgae/GCFGAE/)| 
|AdaFGL: A New Paradigm for Federated Node Classification with Topology Heterogeneity| ICDE| 2024 | [[Paper]](https://arxiv.org/abs/2401.11750) [[Code]](https://github.com/xkLi-Allen/AdaFGL) |
|FedGTA: Topology-aware Averaging for Federated Graph Learning | VLDB | 2024| [[Paper]](https://dl.acm.org/doi/abs/10.14778/3617838.3617842) [[Code]](https://github.com/xkLi-Allen/FedGTA)|
|Federated Graph Learning under Domain Shift with Generalizable Prototypes | AAAI | 2024 |[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29468) [[Code]](https://github.com/GuanchengWan/FGGP) | 
| FedGT: Federated Node Classification with Scalable Graph Transformer| arXiv| 2024| [[Paper]](https://arxiv.org/abs/2401.15203)|  
| FedGL: Federated graph learning framework with global self-supervision| IS | 2024| [[Paper]](https://www.sciencedirect.com/science/article/pii/S002002552301561X) |
| Deep Efficient Private Neighbor Generation for Subgraph Federated Learning| SDM| 2024 | [[Paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611978032.92)|


    
    
</details>


<details>
    <summary> Survey / Library / Benchmarks</summary>
    
| Title | Venue | Year | Materials |
| ----- | ----- | ---- | --------- |
| Federated graph learning--a position paper| arXiv | 2021 | [[Paper]](https://arxiv.org/abs/2105.11099)| 
|FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks | arXiv|2021 | [[Paper]](https://arxiv.org/abs/2104.07145) [[Code]](https://github.com/FedML-AI/FedGraphNN)|
| Federated graph machine learning: A survey of concepts, techniques, and applications| SIGKDD | 2022 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3575637.3575644) |
| Federatedscope-gnn: Towards a unified, comprehensive and efficient package for federated graph learning| KDD| 2022 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3534678.3539112) [[Code]](https://github.com/alibaba/FederatedScope) |
|Federated Graph Neural Networks: Overview, Techniques, and Challenges|TNNLS| 2024 |[[Paper]](https://ieeexplore.ieee.org/abstract/document/10428063)|

</details>


Moreover, we categorize various commonly used graph datasets in recent FGL studies

<details>
    <summary>FGL Datasets</summary>

| Name | Node Feature | Node Label | Edge Feature | Edge Label | Graph Label | Nodes | Edges    | Graphs | Materials|
| ----------- | ------------ | ---------- | ------------ | ---------- | ----------- | ------ | --- | ------ | -------------------- |
| Cora | 1433 | 7          | -            | -       | -           | 2708   | 5429    | 1      | [[Paper]](https://arxiv.org/abs/1603.08861)                                                              |
| Citeseer    | 3703         | 6          | -            | -  | -           | 3327   | 4732   | 1      | [[Paper]](https://arxiv.org/abs/1603.08861)                                                              |
| Pubmed      | 500          | 3          | -            |   -    | -           | 19717  |44338   | 1      | [[Paper]](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157)                        |
| NELL        | 5414         | 210        | -            |  -   | -           | 65755  | 66144   | 1      | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/7519)                                         |
| MUTAG       |             | -          | -            | -      | 2           | 17.93  | 19.79 | 188    | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| BZR         |             | -          | -            | -      | 2           | 35.75  | 38.36  | 405    | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| COX2        |             | -          | -            | -      | 2           | 41.22  | 43.45 | 467    | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| DHFR        |             | -          | -            | -      | 2           | 42.43  | 44.54 | 467    | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| PTC_MR      |             | -          | -            | -     | 2           | 14.29  |14.69  | 344    | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| AIDS        |             | -          | -            | -  | 2           | 15.69  |16.20    | 2000   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| NCI1        |             | -          | -            | -  | 2           | 29.87  |32.30     | 4110   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| ENZYMES     |             | -          | -            |- | 6           | 32.63  | 62.14    | 600    | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| DD          |             | -          | -            | - | 2           | 284.32 |715.66   | 1178   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| PROTEINS    |             | -          | -            | -  | 2           | 39.06  |72.82    | 1113   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| COLLAB      |             | -          | -            | - | 3           | 74.49  | 2457.78    | 5000   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| IMDB-BINARY |             | -          | -            | - | 2           | 19.77  | 96.53      | 1000   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) |
| IMDB-MULTI  |             | -          | -            | - | 3           | 13.00  |65.94      | 1500   | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html)|
| Amazon Photo | 745 | 8 | - | - | - | 7487 | 119043 | 1 | [[Paper]](https://arxiv.org/abs/1811.05868)|
| Amazon Computer | 767 | 10 | - | - | - | 13381 | 245778 | 1 | [[Paper]](https://arxiv.org/abs/1811.05868)|
| Coauthor CS | 6805 | 15 | - | - | - | 18333 | 81894 | 1 | [[Paper]](https://arxiv.org/abs/1811.05868)|
| Coauthor Physics | 8415 | 5 | - | - | - | 34493 | 247962 | 1 | [[Paper]](https://arxiv.org/abs/1811.05868)|
| Chameleon | 2325 | 5 | - | - | - | 2277 | 36101 | 1 | [[Paper]](https://arxiv.org/abs/2002.05287)|
| Squirrel | 2089 | 5 | - | - | - | 5201 | 216933 | 1 | [[Paper]](https://arxiv.org/abs/2002.05287)|
| Tolokers | 10 | 2 | - | - | - | 11758 | 519000 | 1 | [[Paper]](https://arxiv.org/abs/2302.11640)|
| Actor | 931 | 5 | - | - | - | 7600 | 29926 | 1 | [[Paper]](https://arxiv.org/abs/2002.05287)|
| Roman-empire | 300 | 18 | - | - | - | 22662 | 32927 | 1 | [[Paper]](https://arxiv.org/abs/2302.11640)|
| Amazon-rating | 300 | 5 | - | - | - | 24492 | 93050 | 1 | [[Paper]](https://arxiv.org/abs/2302.11640)|
| Ogbn-arxiv | 128 | 40 | - | - | - | 169343 | 2315598 | 1 | [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html)|
| Ogbn-products | 100 | 47 | - | - | - | 2449029 | 61859140 | 1 | [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html)|
| Genius |  | 2 | - | - | - | 421961 | 922868 | 1 | [[Paper]](https://ojs.aaai.org/index.php/ICWSM/article/view/18068)|
| DBLP | - | 4 | - | - | - | 26128 | 239566 | 1 | [[Paper]](https://openreview.net/forum?id=Qs81lLhOor)|
| ACM | - | 3 | - | - | - | 10942 | 547872 | 1 | [[Paper]](https://openreview.net/forum?id=Qs81lLhOor)|
| IMDB | - | 5 | - | - | - | 21420 | 86642 | 1 | [[Paper]](https://openreview.net/forum?id=Qs81lLhOor)|
| Freebase | - | 7 | - | - | - | 180098 | 1057688 | 1 | [[Paper]](https://openreview.net/forum?id=Qs81lLhOor)|
| Ogbn-mag | - | 349 | - | - | - | 1939743 | 21111007 | 1 | [[Paper]](https://arxiv.org/abs/2103.09430)|
| DBLP-dm | 200 | 12 | - | - | - | 46582 | 7097924 | 1 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583471)|
| PubMed-diabetes | 200 | 3 | - | - | - | 13778 | 588529 | 1 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583471)|
| NELL | 2792 | 5 | - | - | - | 41671 | 39250315 | 1 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583471)|
| MIMIC3  | 6671 | 6 | - | - | - | 58495 | 30603469 | 1 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583471)|
| ACMv9  | 7537 | 6 | - | - | - | 7410 | 11135 | 1 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3366423.3380219)|
| DBLPv8  | 7537 | 6 | - | - | - | 5578 | 7341 | 1 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3366423.3380219)|
| Twitch  |  |  | - | - | - | 168114 | 6797557 | 1 | [[Paper]](https://arxiv.org/abs/2101.03091)|
| FB15K-237  | - | - | - | 237 | - | 14541 | 310116 | 1 | [[Paper]](https://aclanthology.org/D15-1174/)|
| WN18RR  | - | - | - | 11 | - | 40943 | 93003 | 1 | [[Paper]](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)|
| NELL-995  | - | - | - | 200 | - | 75492 | 154213 | 1 | [[Paper]](https://arxiv.org/abs/1707.06690)|
| Name | Node Feature | Node Label | Edge Feature | Edge Label | Graph Label | Nodes | Edges    | Graphs | Materials|
| FedDBLP  |  | 4 | - | 200 | - | 52202 | 271054 | 1 | [[Paper]](https://arxiv.org/abs/2204.05562)|
| CSR  |  |  | - |  | - |  |  | 1 | [[Paper]](https://arxiv.org/abs/2204.05562)|
| Wiki-CS  | 300 |  10 | - |  | - | 11701 | 216123 | 1 | [[Paper]](https://arxiv.org/abs/2007.02901)|
| CoraFull  | 8710 | 70 | - |  | - | 19793 | 65311 | 1 | [[Paper]](https://arxiv.org/abs/1707.03815)|


</details>

## Get Started
You can modify the experimental settings in `/config.py` as needed, and then run `/main.py` to start your work with OpenFGL. Moreover, we provide various configured jupyter notebook examples, all of which can be found in `/examples`.

### Scenario and Dataset Simulation Settings

```python
--scenario           # fgl scenario
--root               # root directory for datasets
--dataset            # list of used dataset(s)
--simulation_mode    # strategy for extracting FGL dataset from global dataset
```

### Communication Settings

```python
--num_clients        # number of clients
--num_rounds         # number of communication rounds
--client_frac        # client activation fraction
```

### FL/FGL Algorithm Settings
```python
--fl_algorithm       # used fl/fgl algorithm
```

### Model and Task Settings
```python
--task               # downstream task
--train_val_test     # train/validatoin/test split proportion
--num_epochs         # number of local epochs
--dropout            # dropout
--lr                 # learning rate
--optim              # optimizer
--weight_decay       # weight decay
--model              # gnn backbone
--hid_dim            # number of hidden layer units
```


### Evaluation Settings

```python
--metrics            # performance evaluation metric
--evaluation_mode    # personalized evaluation / global evaluation
```
## Cite
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
```
```
