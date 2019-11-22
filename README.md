# GraphRec-WWW19

## GraphRec: Graph Neural Networks for Social Recommendation

This is our implementation for the paper:

[Wenqi Fan](https://wenqifan03.github.io), Yao Ma , Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. [Graph Neural Networks for Social Recommendation](https://arxiv.org/pdf/1902.07243.pdf). 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. 
Preprint[https://arxiv.org/abs/1902.07243]


## Abstract
In recent years, Graph Neural Networks (GNNs), which can naturally integrate node information and topological structure, have been demonstrated to be powerful in learning on graph data. These advantages of GNNs provide great potential to ad- vance social recommendation since data in social recommender systems can be represented as user-user social graph and user-item graph; and learning latent factors of users and items is the key. However, building social recommender systems based on GNNs faces challenges. For example, the user-item graph encodes both interactions and their associated opinions; social relations have heterogeneous strengths; users involve in two graphs (e.g., the user-user social graph and the user-item graph). To address the three aforementioned challenges simultaneously, in this paper, we present a novel graph neural network framework (GraphRec) for social recommendations. In particular, we provide a principled approach to jointly capture interactions and opinions in the user-item graph and propose the framework GraphRec, which coherently models two graphs and heterogeneous strengths. Extensive experiments on two real-world datasets demonstrate the effectiveness of the proposed framework GraphRec.



## Introduction
 Graph Data in Social Recommendation. It contains two graphs including the user-item graph (left part) and the user-user social graph (right part). Note that the number on the edges of the user-item graph denotes the opinions (or rating score) of users on the items via the interactions.
![ 123](intro.png "Social Recommendations")


## Our Model GraphRec
The overall architecture of the proposed model. It contains three major components: user modeling, item modeling, and rating prediction.The first component is user modeling, which is to learn latent factors of users. As data in social recommender systems includes two different graphs, i.e., a social graph and a user-item graph, we are provided with a great opportunity to learn user representations from different perspectives. Therefore, two aggregations are introduced to respectively process these two different graphs. One is item aggregation, which can be utilized to understand users via interactions between users and items in the user-item graph (or item-space). The other is social aggregation, the relationship between users in the social graph, which can help model users from the social perspective (or social-space). Then, it is intuitive to obtain user latent factors by combining information from both item space and social space. The second component is item modeling, which is to learn latent factors of items. In order to

![ 123](GraphRec.png "GraphRec")


## Code

Author: Wenqi Fan (https://wenqifan03.github.io, email: wenqifan03@gmail.com) 

Also, I would be more than happy to provide a detailed answer for any questions you may have regarding GraphRec.

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

## Environment Settings
python: 3.6
pytorch: >0.2

## Example to run the codes

Run GraphRec:
```
python run_GraphRec_example.py
```

# Acknowledgements
The original version of this code base was from GraphSage. We owe many thanks to William L. Hamilton for making his code available. 
Please see the paper for funding details and additional (non-code related) acknowledgements.

Last Update Date: Oct, 2019
