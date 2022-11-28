# Data

## Data processing script
We provide processing scripts for data processing. This would also enable running our graph-augmented generative framework on your own data. The bash script will take several hours for processing SEESAW.

```shell script
$ bash data_processing.sh
```

Note, to successfully run ```step_1_entity_extraction.py```, you should obtain Google Cloud Credential first. See [here](https://cloud.google.com/docs/authentication/provide-credentials-adc) for more details. Afterwards, replace ```"PATH_TO_YOUR_GOOGLE_CLOUD_CREDETIAL"``` (line 36 in ```step_1_entity_extraction.py```) with the actual path to the credential file


## Processed data
We also provide the processed version of SEESAW datasets, upon which you can directly run our models. Processed textual data is stored under [```processed```](./processed) directory, and processed graphical data is stored [here](https://drive.google.com/drive/folders/1vaBdJZCFPnQXgs1xj-C0NVTUNRmacnpk?usp=sharing). Please move textual data to this directory (i.e., ```PATH_TO_SEESAW/SEESAW/data```) and download graphical data from google drive to [```graph```](./graph) directory before running models. 

## IdeologyKB
IdeologyKB is a knowledge base derived from [Voteview](https://voteview.com/) revealing the ideological position of major U.S. politicians. Key is U.S. politicians' canonical names based on Wikipedia. Values include registered affiliation, full name (incl. first, last, middle and suffic, if possible), last year in office (as of 2022), first and second principal dimensions estimated by DW-NOMINATE. Note that under the context of American political spectrum, the first principal component can usually be interpreted as "liberal" vs. "conservative" (also referred to as "left" vs. "right").

Please cite the following two work if you use our generated ideologyKB:
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
```
@misc{voteview,
  title={Voteview: Congressional Roll-Call Votes Database},
  author={Jeffrey Lewis and Keith Poole and Howard Rosenthal and Adam Boche and Aaron Rudkin and Luke Sonnet},
  year={2022}
}
```