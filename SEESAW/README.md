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

For more details about SEESAW, please check out Section 3 (SEESAW Collection and Annotation) and Appendix A (Annotation Guideline) in our paper, where we detail the semantics of each field we annotated.

#### "article ideology" field in ```annotations``` v.s. "bias_label" field in ```meta```: 

- "article ideology" indicates our annotators' 5-way estimation of the ideology of the media organization that published this article.

- "bias_label" are bias labels annotated by [AllSides](https://www.allsides.com/media-bias/media-bias-chart) (as of June 2022) in 5-way. 0: (Far) Left, 1: Lean Left, 2: Center, 3: Lean Right, 4: (Far) Right.


## Processed data
We also provide data processing script and processed data under [data](../data) directory.
