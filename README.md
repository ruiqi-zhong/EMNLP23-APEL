# EMNLP23-APEL

Code and data for the paper: [Non-Programmers Can Label Programs Indirectly via Active Examples: A Case Study with Text-to-SQL](https://arxiv.org/abs/2205.12422), to appear in EMNLP 2023. Joint work with Charlie Snell, Dan Klein, and Jason Eisner.

To install: run ```pip3 install -r requirements.txt```

```example_minimal_witness.py``` contains a simple example for runing the algorithm in Section 3 which maximize the information gain subject to a size constraint. In this algorithm, we first randomly generate large random databases, choose the most informative one, then reduce its size.

```spiderdevfixes.csv``` contains all the original SPIDER annotations we corrected, along with the reason. The corresponding author of the SPIDER dataset endorses our correction.

Citation:

```
@InProceedings{Zhong-Snell-Klein-Eisner:2023:APEL,
  title     = {Non-Programmers Can Label Programs Indirectly via Active Examples: A Case Study with Text-to-SQL},
  author    = {Ruiqi Zhong and Charlie Snell and Dan Klein and Jason Eisner},
  booktitle = {Proceedings of EMNLP},
  address   = {},
  pages     = {},
  month     = {December},
  year      = {2023},
}
```
