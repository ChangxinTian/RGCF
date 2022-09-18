# Robust Graph Collaborative Filtering (RGCF)

This is our Pytorch implementation for the paper:

> Changxin Tian, Yuexiang Xie, Yaliang Li, Nan Yang and Wayne Xin Zhao. "Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering." SIGIR 2022.

## Introduction
Recently, graph neural networks (GNN) have been successfully applied to recommender systems as an effective collaborative filtering (CF) approach. 
However, existing GNN-based CF models suffer from noisy user-item interaction data, which seriously affects their effectiveness and robustness in 
real-world applications. Although there have been several studies on data denoising in recommender systems, they either neglect direct intervention 
of noisy interaction in the message-propagation of GNN, or fail to preserve the diversity of recommendation when denoising.
To tackle the aforementioned issues, in this paper, we propose a novel GNN-based CF model, named Robust Graph Collaborative Filtering (denoted as RGCF), 
to denoise unreliable interactions for recommendation. Specifically, RGCF consists of a graph denoising module and a diversity preserving module. 
The graph denoising module is designed for reducing the impact of noisy interactions in the representation learning of GNN, by adopting both a hard 
denoising strategy (i.e., discarding interactions that are confidently estimated as noise) and a soft denoising strategy (i.e., assigning reliability 
weights for each remaining interaction). In the diversity preserving module, we build up a diversity augmented graph and propose an auxiliary 
self-supervised task based on mutual information maximization (MIM) for enhancing the denoised representation and preserving the diversity of 
recommendations. These two modules are integrated in a multi-task learning manner that jointly improves the recommendation performance of the 
proposed RGCF. We conduct a series of experiments on three real world recommendation datasets and three synthesized datasets to demonstrate the 
effectiveness of the proposed RGCF. The experiment results show that RGCF is more robust against noisy interactions and achieves significant improvement 
compared with baseline models in terms of accuracy and diversity of recommendation.

## Requirements:
* Python=3.7.10
* PyTorch=1.7.0
* cudatoolkit=10.1
* pandas=1.3.2
* numpy=1.21.2
* recbole=1.0.0

## Dataset:

For MovieLens-1M, you can execute

```
cd dataset
unzip ml-1m.zip -d ./ml-1m
```

For Yelp and Amazon-Book, you can download from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets).


## Training:

```
# run on the real-world datasets
python -u run_rgcf.py

# run on the synthetic datasets
python -u run_rgcf.py --ptb_strategy=replace
```

## Reference:
Any scientific publications that use our codes should cite the following paper as the reference:

 ```
 @inproceedings{tian2022learning,
    author = {Changxin Tian and Yuexiang Xie and Yaliang Li and Nan Yang and Wayne Xin Zhao},
    title = {Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering.},
    booktitle = {{SIGIR}},
    year = {2022},
 }
 ```

If you have any questions for our paper or codes, please send an email to tianchangxin@ruc.edu.cn.
