
![1301717130101_ pic](https://github.com/zyl24/OpenFGL/assets/59046279/e21b410f-2b5d-4515-8ab5-a176f98805a7)


# Open Federated Graph Learning (OpenFGL)
OpenFGL is a comprehensive, user-friendly algorithm library, complemented by an integrated evaluation platform, designed specifically for researchers in the field of federated graph learning (FGL).

<p align="center">
  <a href="https://arxiv.org/abs/2408.16288">Paper</a> •
  <a href="#Library Highlights">Highlights</a> •
  <a href="https://pypi.org/project/openfgl-lib/1.0.0/">Installation</a> •
  <a href="https://openfgl.readthedocs.io/en/latest/">Docs</a> •
  <a href="#Citation">Citation</a> 
</p>



[![Stars](https://img.shields.io/github/stars/zyl24/OpenFGL.svg?color=orange)](https://github.com/zyl24/OpenFGL/stargazers) ![](https://img.shields.io/github/last-commit/zyl24/OpenFGL) 
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992) -->

 



## Highlights

- 2 FGL Scenarios: Graph-FL and Subgraph-FL
- 10+ FGL Algorithms
- 34 FGL Datasets
- 12 GNN Models
- 5 Downstream Tasks
- Comprehensive FGL Data Property Analysis

## Get Started

```python
import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "your_root_path"

args.dataset = ["Cora"]

args.model = ["gcn"]

args.metrics = ["accuracy"]

args.fl_algorithms = "fedavg"

trainer = FGLTrainer(args)

trainer.train()
```


## Citation
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
```
@misc{li2024openfglcomprehensivebenchmarksfederated,
      title={OpenFGL: A Comprehensive Benchmarks for Federated Graph Learning}, 
      author={Xunkai Li and Yinlin Zhu and Boyang Pang and Guochen Yan and Yeyu Yan and Zening Li and Zhengyu Wu and Wentao Zhang and Rong-Hua Li and Guoren Wang},
      year={2024},
      eprint={2408.16288},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.16288}, 
}
```
