# Role-Aware Modeling for N-ary Relational Knowledge Bases

This repository is the official implementation of our WWW'2021 paper "Role-Aware Modeling for N-ary Relational Knowledge Bases".

## Requirements

To install requirements:

```setup
python 3.7.4
pytorch 1.1
```


## Running a model

To train (and evaluate) the model in the paper, run this command:

```
python main.py --dataset FB-AUTO --num_iterations 200 --batch_size 64 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4 --eval_step 1 --valid_patience 10 -ary 2 -ary 4 -ary 5
```

>📋 The ary append for WikiPeople is 
```
-ary 2 -ary 3 -ary 4 -ary 5 -ary 6 -ary 7 -ary 8 -ary 9
```
>📋 The ary append for JF17K is 
```
-ary 2 -ary 3 -ary 4 -ary 5 -ary 6
```
>📋 The ary append for WN18/FB15k  is 
```python
-ary 2
```

>Evaluation interval is determined by the parameter "eval_step"


## Hyperparameters Settings

|  Dataset   |  d   |  K   |  lr   |  dr   | drop_role | drop_ent | batch_size |
| :--------: | :--: | :--: | :---: | :---: | :-------: | :------: | :--------: |
| WikiPeople |  25  |  10  | 0.003 | 0.995 |    0.0    |   0.2    |     64     |
|   JF17K    |  50  |  10  | 0.005 | 0.995 |    0.2    |   0.4    |     64     |
|  FB-AUTO   |  50  |  10  | 0.005 | 0.995 |    0.2    |   0.4    |     64     |
|    WN18    |  50  |  10  | 0.002 | 0.995 |    0.0    |   0.4    |    128     |
|   FB15k    | 100  |  50  | 0.001 | 0.99  |    0.2    |   0.0    |    128     |

## Results

Our model achieves the following performance on WikiPeople, JF17K, FB-AUTO, WN18, and FB15k.

|  Dataset   |  MRR  | Hits@10 | Hits@1 |
| :--------: | :---: | :-----: | :------: |
| WikiPeople | 0.380 |  0.541  | 0.278  |
|   JF17K    | 0.539 |  0.690  | 0.463  |
|  FB-AUTO   | 0.830 |  0.876  | 0.803  |
|    WN18    | 0.947 |  0.952  | 0.943  |
|   FB15k    | 0.803 |  0.882  | 0.756  |


## Reference
```latex
@inproceddings{liu2021ram,
	title 	  = {Role-Aware Modeling for N-ary Relational Knowledge Bases},
	author	  = {Liu, Yu and Yao, Quanming and Li, Yong},
	booktitle = {The World Wide Web Conference},
	year      = {2021},
}
```