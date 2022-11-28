import re
import os
import numpy as np  
import csv
import os
import json
from tqdm import tqdm
from google.cloud import language_v1
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from os import listdir
import spacy
import time
import ast

nlp = English()
nlp = spacy.load("en_core_web_sm")

label_look_up = {'NEG':'negative', 'POS':'positive'}

SPAN = 3

files = listdir("../../SEESAW/meta")
train_triplets = []
valid_triplets = []
test_triplets = []

for file in files:
    with open("../../SEESAW/meta/"+file, "r") as f:
        data = json.load(f)
        if data["Split"] == "train":
            train_triplets.append(data["triplet_id"])
        elif data["Split"] == "valid":
            valid_triplets.append(data["triplet_id"])
        elif data["Split"] == "test":
            test_triplets.append(data["triplet_id"])

articleDir = "../../SEESAW/articles"
annotationDir = "../../SEESAW/annotations"
files = listdir(articleDir)
files.sort()


train_data = []
valid_data = []
test_data = []

def construct_context(article, idx):
    sentence_list = [x["text"] for x in article if x["text"].strip()!=""]
    idx2sentences = {}
    for x in article:
        if x["text"].strip()!="":
            idx2sentences[x["id"]] =  len(idx2sentences)

    assert len(idx2sentences) == len(sentence_list)

    sentence_idx = idx2sentences[idx]
    prior = " ".join(sentence_list[max([sentence_idx-SPAN,0]) : sentence_idx])
    target = sentence_list[sentence_idx]
    after = " ".join(sentence_list[sentence_idx+1 : sentence_idx+1+SPAN])

    prior = " ".join([word.strip() for word in prior.split()])
    target = " ".join([word.strip() for word in target.split()])
    after = " ".join([word.strip() for word in after.split()])

    return prior.strip(), target.strip(), after.strip()

for file in files:
    triplet_id = file.split(".")[0].split("_")[1]
    with open(os.path.join(articleDir,file), 'r') as article_file, open(os.path.join(annotationDir,file), 'r') as annotation_file:
        article_json = json.load(article_file)
        annotation_json = json.load(annotation_file)
        article = article_json["text"]

        document_text = " ".join([word.strip() for x in article for word in x["text"].split()])
        document_text = document_text.strip()
        assert "  " not in document_text
        doc_entities_list = [x[0].strip() for x in annotation_json["entity lists"]]

        for idx, item in annotation_json["sentence-level stances"].items():
            prior, target, after = construct_context(article, int(idx))
            context_text = prior + " " + target + " " + after
            context_text = context_text.strip()
            text = target.strip()
            assert text in context_text
            assert context_text in document_text

            entities = [({"ent":"source", "text":" ".join([word.strip() for word in x["subject"].split()]).strip()}, {"ent":"target", "text":" ".join([word.strip() for word in x["object"].split()]).strip()}, {"label":label_look_up[x["sentiment"]]}) for x in item]
            item_entities_list = set()
            for x in entities:
                ent1 = x[0]["text"]
                ent2 = x[1]["text"]
                if ent1 not in ["None", "Not in the list"]:
                    item_entities_list.add(ent1.strip())
                if ent2 not in ["None", "Not in the list"]: 
                    item_entities_list.add(ent2.strip())
            item_entities_list = list(item_entities_list)
            unique_id = file.split(".")[0] + "/" + str(idx)
            if triplet_id in train_triplets:
                train_data.append({"text":text, "entities":entities, "context text":context_text, "document text":document_text, "item entities list":item_entities_list, "entities list":doc_entities_list, "unique_id":unique_id })
            elif triplet_id in valid_triplets:
                valid_data.append({"text":text, "entities":entities, "context text":context_text, "document text":document_text, "item entities list":item_entities_list, "entities list":doc_entities_list, "unique_id":unique_id })
            elif triplet_id in test_triplets:
                test_data.append({"text":text, "entities":entities, "context text":context_text, "document text":document_text, "item entities list":item_entities_list, "entities list":doc_entities_list, "unique_id":unique_id })





with open('../SEESAW_data.train.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4) 
    print(len(train_data))               
with open('../SEESAW_data.valid.json', 'w') as outfile:
    json.dump(valid_data, outfile, indent=4)  
    print(len(valid_data))
with open('../SEESAW_data.test.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4)  
    print(len(test_data))
