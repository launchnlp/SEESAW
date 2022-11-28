from distutils import core
import os
from pickle import MEMOIZE
from re import M
from tqdm import tqdm
from allennlp.predictors import Predictor
import allennlp_models.tagging
from collections import Iterable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()
stopwords_en = stopwords.words('english')

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt



Possessive_adjectives = ["his","her","their","my","your","our", "its"]
# Imported from https://gist.github.com/mohataher/837a1ed91aab7ab6c8321a2bae18dc3e
Pronoun_List=['i','you','my','mine','myself''we','us','our','ours','ourselves''you','you','your','yours','yourself''you','you','your','your','yourselves''he','him','his','his','himself''she','her','her','her','herself''it','it','its','itself''they','them','their','theirs','themself''they','them','their','theirs','themselves', 'this', 'that', 'these', 'those']

NUM=2
openie_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", cuda_device=NUM)
srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", cuda_device=NUM)
coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz", cuda_device=NUM)



def coref_resolved(doc):
    results=coref_predictor.predict_tokenized(doc.split())
    document = results['document']
    sentence = document.copy()

    for x,y in zip(doc.split(), sentence):
        if x!= y:
            print(x)
            print(y)
    assert doc == " ".join(sentence)
    indices = [x for x in range(len(document))]
    indices_matching = {x:-1 for x in indices}
    if len(document) != len(doc.split()):
        return None, None, None
    assert len(document) == len(doc.split())
    # 
    clusters = results['clusters']
    cluster_mention_name = {}
    for cluster in clusters:
        tmp_mention = []
        tmp_name = None
        for x in cluster:
            tmp_mention.append([x[0],x[1]+1])
            tokens = " ".join(document[x[0]:x[1]+1])

            # remove 's and stop words
            if tokens.split()[-1] == "'s" or tokens.split()[-1] =="\u2019s":
                tokens = tokens.replace("'s", "")
                tokens = tokens.replace("\u2019s", "")
            tokens_cmp = " ".join([w for w in tokens.split() if not w.lower() in stopwords_en])
            if tmp_name == None:
                tmp_name = tokens
            else:
                tmp_name_cmp = " ".join([w for w in tmp_name.split() if not w.lower() in stopwords_en])
                if len(tokens_cmp) > len(tmp_name_cmp):
                    tmp_name = tokens 
        cluster_mention_name[tmp_name] = tmp_mention
    return " ".join(sentence), cluster_mention_name

def triplet_extraction(doc):
    results = srl_predictor.predict_tokenized(doc.split())
    words = results["words"]
    triples = []
    triples_idx = []

    if len(words) != len(doc.split()):
        return None, None, None
    for triple in results["verbs"]:
        verb = triple['verb']
        verb = lemmatizer.lemmatize(verb, 'v')
        tags = triple['tags']
        arg0 = None
        arg1 = None
        arg2 = None
        if "B-ARG0" in tags:
            start0 = tags.index("B-ARG0")
            if "I-ARG0" in tags:
                second_start0 = tags.index("I-ARG0")
                if start0 > second_start0:
                    continue
            length0 = np.sum(np.array(tags)=="I-ARG0")
            arg0 = " ".join(words[start0:start0+length0+1])
        if "B-ARG1" in tags:
            start1 = tags.index("B-ARG1")
            if "I-ARG1" in tags:
                second_start1 = tags.index("I-ARG1")
                if start1 > second_start1:
                    continue            
            length1 = np.sum(np.array(tags)=="I-ARG1")
            arg1 = " ".join(words[start1:start1+length1+1])
        if "B-ARG2" in tags:
            start2 = tags.index("B-ARG2")
            if "I-ARG2" in tags:
                second_start2 = tags.index("I-ARG2")
                if start2 > second_start2:
                    continue            
            length2 = np.sum(np.array(tags)=="I-ARG2")
            arg2 = " ".join(words[start2:start2+length2+1])
        if "B-V" in tags:
            start_v = tags.index("B-V")
            if "I-V" in tags:
                second_startv = tags.index("I-V")
                if start_v > second_startv:
                    continue                 
            length_v = np.sum(np.array(tags)=="I-V")
            
            if arg0 != None and arg1 != None:
                triples.append((arg0, verb, arg1))
                triples_idx.append(([start0, start0+length0+1],[start_v, start_v+length_v+1],[start1, start1+length1+1]))
            if arg0 != None and arg2 != None:
                triples.append((arg0, verb, arg2))
                triples_idx.append(([start0, start0+length0+1],[start_v, start_v+length_v+1],[start2, start2+length2+1]))   
            if arg1 != None and arg2 != None:
                triples.append((arg1, verb, arg2))
                triples_idx.append(([start1, start1+length1+1],[start_v, start_v+length_v+1],[start2, start2+length2+1]))      
    return triples, triples_idx

def sentence_process(doc):
    triples, triples_idx = triplet_extraction(doc)
    if triples == None:
        return None, None, None, None
    coref_doc, cluster_mention_name = coref_resolved(doc)
    if coref_doc == None:
        return None, None, None, None 

    return triples, triples_idx, coref_doc, cluster_mention_name