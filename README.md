# CoRA
Codes and datasets for our paper "Improving Long-Tail Relation Extraction with Collaborating Relation-Augmented Attention"

If you use the code, please cite the following [paper](https://arxiv.org/pdf/2010.03773.pdf):

```
@article{li2020improving,
  title={Improving Long-Tail Relation Extraction with Collaborating Relation-Augmented Attention},
  author={Li, Yang and Shen, Tao and Long, Guodong and Jiang, Jing and Zhou, Tianyi and Zhang, Chengqi},
  journal={arXiv preprint arXiv:2010.03773},
  year={2020}
}
```

## Requirements

The model is implemented using tensorflow. The versions of packages used are shown below.

* pytorch = 1.4.0
* numpy = 1.19.2
* scipy = 1.5.2

## Data preparation

First unzip the `./raw_data/data.zip` and put all the files under `./raw_data`. Once the original raw text corpus data is in `./raw_data`.

## Train the model
For CoRA,

    python main_CoRA.py --is_training True

## Evaluate the model

Run various evaluation by specifying `--mode` in commandline, see the paper for detailed description for these evaluation methods.

    python main_CoRA.py --mode [test method: pr, pone, ptwo, pall, hit_k_100, hit_k_200]

## Pretrained models

The pretrained models is already saved at `./outputs/ckpt/`.

    python main_CoRA.py --mode [test method: pr, pone, ptwo, pall, hit_k_100, hit_k_200] --test_pretrained

