# Generative Entity-to-Entity Stance Detection with Knowledge Graph Augmentation

## Introduction
This repository contains codes and dataset for paper ["Generative Entity-to-Entity Stance Detection with Knowledge Graph Augmentation"](https://aclanthology.org/2022.emnlp-main.676/) (In <em>Proceeding of the 2022 Conference on Empirical Methods in Natural Language Processing</em>).

## Set up
Run the following commands to clone the repository.
```shell script
$ git clone https://github.com/launchnlp/SEESAW.git
```

Before running our codes, please run the following script to have all dependencies set up.
```shell script
$ bash requirements.sh
```

## Data 
Raw SEESAW can be found under [SEESAW](./SEESAW) directory. Please read README under [SEESAW](./SEESAW) directory for more information.

Processed data and the script for data processing can be found under [data](./data) directory.

For the data used for Task B: Stance-only prediction for pairwise entities. (see more details in Section 5.1 in our paper), please download from the [original data source](https://github.com/bywords/directed_sentiment_analysis).


## Experiments: Generative Entity-to-Entity Stance Detection
*We are still refactoring and cleaning the codes,. Please stay tuned for more updates.*

## Citation
Please cite our paper if you use our **codes** and/or **SEESAW** dataset:
```
@inproceedings{zhang-etal-2022-seesaw,
    title = "Generative Entity-to-Entity Stance Detection with Knowledge Graph Augmentation",
    author = "Zhang, Xinliang Frederick  and
      Beauchamp, Nicholas  and
      Wang, Lu",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
Please also cite the following paper if you run **POLITICS** as your backbone model:
```
@inproceedings{liu-etal-2022-politics,
    title = "{POLITICS}: Pretraining with Same-story Article Comparison for Ideology Prediction and Stance Detection",
    author = "Liu, Yujian  and
      Zhang, Xinliang Frederick  and
      Wegsman, David  and
      Beauchamp, Nicholas  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: {NAACL} 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    pages = "1354--1374",
}
```

## Contact
If you have any question, please contact Xinliang Frederick Zhang ```<xlfzhang@umich.edu>``` or create a Github issue.
