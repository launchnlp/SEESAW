# SEESAW Dataset

## TL;DR
SEESAW is a large-scale dataset with **10,619** stance annotations labeled at the sentence-level from **609** news articles (i.e., **203** news stories/triplets) of different ideological leanings.

## Directory organization

- ```./annotations``` directory contains annotation for 609 news articles where each article is coded by one or more than one annotator from our 6-people annotator pool, indexed from ```A``` to ```F```.

- ```./articles``` directory contains 609 news articles where articles have been segmented into sentences.

- ```./meta``` directory contains meta information for each news story/triplet.


## Addtional basic information
SEESAW coveres **1,757** distinct entities. **62.4%** of the stance triplets have negative sentiment.  

SEESAW also supports other studies besides the stance triplet extraction: 1) article-level main/salient entity identification, 2) article-level entity-based sentiment classfication, 3) entity-based ideology prediction, 4) article-level ideological leaning prediction. 

For more details about SEESAW, please check out Section 3 (SEESAW Collection and Annotation) in our paper.

## Processed data
We also provide data processing script and processed data under [data](../data) directory.