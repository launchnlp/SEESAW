import json
import csv
import os

from tqdm import tqdm
from nltk.corpus import stopwords
import spacy
import requests
import string
import re
import statistics
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()
nlp = spacy.load("en_core_web_sm")


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def process_data(file_name, db, special_treatment=False):
    # special_treatment only matters for entity_ordering since it determines all possible entities in an article 
    # train/valid: all auto-extracted entities + human-coded entities
    # test: all auto-extracted entities
    new_data = []
    data_file = '../SEESAW_data.'+file_name+'.json'
    entity_automatched_file = "../cache/data.entities_automatch."+file_name+".json"
    entity2mention_file = "../cache/data.entities2mentions."+file_name+".json"

    with open(data_file, 'r') as file, open(entity_automatched_file) as file_convertor, open(entity2mention_file) as file_ent2mention:
        data = json.load(file)
        data_matcher = json.load(file_convertor)
        data_ent2mention = json.load(file_ent2mention)
        assert len(data) == len(data_matcher)
        assert len(data) == len(data_ent2mention)

        for item, item_matcher, item_ent2mention in tqdm(zip(data, data_matcher, data_ent2mention)):
            new_item_entities = []
            auto2gt_item = {} # key: auto entity, value: GT annotation
            auto2gt = {} # key: auto entity, value: GT annotation
            for gt in item_matcher:
                auto = item_matcher[gt][0]
                if auto not in auto2gt:
                    auto2gt[auto] = [gt]
                else:
                    auto2gt[auto].append(gt)
            for gt in item["item entities list"]:
                auto  = item_matcher[gt][0]
                if auto not in auto2gt_item:
                    auto2gt_item[auto] = [gt]
                else:
                    auto2gt_item[auto].append(gt)
            for x in auto2gt:
                auto2gt[x].sort()
            for x in auto2gt_item:
                auto2gt_item[x].sort()
            
            # update text, context and doc text
            text = item['text']
            context_text = item['context text']
            document_text = item['document text']

            doc = nlp(text)
            tokens = list(doc)
            text = " ".join([str(x) for x in tokens])

            doc = nlp(context_text)
            tokens = list(doc)
            context_text = " ".join([str(x) for x in tokens]) 

            doc = nlp(document_text)
            tokens = list(doc)
            document_text = " ".join([str(x) for x in tokens])  

            item['text'] = text
            item['context text'] = context_text         
            item['document text'] = document_text


            # obtain basic info for sorting
            doc_length = len(document_text.split())
            text_start_char = document_text.index(text)
            text_start = len(document_text[0:text_start_char].split())
            text_end = text_start + len(text.split())
            text_length = len(text.split())
            context_length = len(context_text.split())

            text_start_char_context = context_text.index(text)
            text_start_context = len(context_text[0:text_start_char_context].split())
            left_context_length = text_start_context  # actually, it's the summation of left context length and text length
            right_context_length = context_length - left_context_length - text_length

            for entity_pair in item["entities"]:
                ent1 = entity_pair[0]
                ent2 = entity_pair[1]  
                label = entity_pair[2]      
                ent1_text_orig = ent1["text"]
                ent2_text_orig = ent2["text"]

                if ent1_text_orig == "None":
                    ent1_text_orig = "<Author>"
                if ent2_text_orig == "None":
                    ent2_text_orig = "<Author>"
                if ent1_text_orig == "Not in the list":
                    ent1_text_orig = "<Someone>"
                if ent2_text_orig == "Not in the list":
                    ent2_text_orig = "<Someone>"

                ent1["auto_match"] = item_matcher[ent1_text_orig][0] if ent1_text_orig not in ["<Author>", "<Someone>"] else ent1_text_orig
                ent2["auto_match"] = item_matcher[ent2_text_orig][0] if ent2_text_orig not in ["<Author>", "<Someone>"] else ent2_text_orig
                
                ent1['wiki'] = item_matcher[ent1_text_orig][1] if ent1_text_orig not in ["<Author>", "<Someone>"] else ""
                ent2['wiki'] = item_matcher[ent2_text_orig][1] if ent2_text_orig not in ["<Author>", "<Someone>"] else ""

                # # Party look-up
                # use canonical name to look up
                ent1["party"] = db[ent1['text']] if ent1['text'] in db else "unknown"
                ent2["party"] = db[ent2['text']] if ent2['text'] in db else "unknown"

                ent1["text"] = ent1_text_orig
                ent2["text"] = ent2_text_orig

                new_item_entities.append((ent1,ent2,label))

            occurences_start = {x: np.array([span[2][0] for span in item_ent2mention[x][0]]) for x in item_ent2mention}
            occurences_end = {x: np.array([span[2][1] for span in item_ent2mention[x][0]]) for x in item_ent2mention}
            entities_ordering = {}

            for x,y in zip(occurences_start, occurences_end):
                assert x ==y
                item_occurences_start = occurences_start[x] - text_start   # entities only appearing in left context have negative value
                item_occurences_end = text_end - occurences_end[y]         # entities only appearing in right context have negative value
                distance_start = np.min(item_occurences_start[item_occurences_start>=0]) if len(item_occurences_start[item_occurences_start>=0]) >0 else 1000000
                distance_end = np.min(item_occurences_end[item_occurences_end>=0]) if len(item_occurences_end[item_occurences_end>=0]) >0 else 1000000

                # entities in text
                if distance_start + distance_end < text_length:
                    entities_ordering[x] = distance_start
                    assert entities_ordering[x] < text_length
                else:
                    # entities in left context
                    if distance_end < left_context_length + text_length:
                        # avoid some weird errors like "FOX NEWS MIDTERM ELECTIONS HEADQUARTERS Trump has"
                        if distance_end < text_length:
                            distance_end = text_length
                            print(x)
                        entities_ordering[x] = distance_end
                        
                        assert entities_ordering[x] >= text_length
                        assert entities_ordering[x] < left_context_length + text_length
                    # entities in right context
                    elif distance_start < text_length + right_context_length:
                        if distance_start < text_length:
                            distance_start = text_length
                            print(x)                        
                        entities_ordering[x] = distance_start + left_context_length 
                        assert entities_ordering[x] >= left_context_length + text_length
                        assert entities_ordering[x] < context_length 
                    # entities not appearing in context
                    else:
                        entities_ordering[x] = 1000000

            entities_ordering = dict(sorted(entities_ordering.items(), key=lambda item: item[1]))
            
            new_entities_ordering = dict()
            for k,v in entities_ordering.items():
                new_entities_ordering[k] = v
                # special treatment on test set 
                if special_treatment is not True:
                    if k in auto2gt_item:
                        for t in auto2gt_item[k]:
                            new_entities_ordering[t] = v 
            
            entities_ordering = new_entities_ordering
            item["entities ordering"] = {k:str(v) for k,v in entities_ordering.items()}

            entities_ordering = list(entities_ordering.keys())

            entity_list = [item_matcher[x][0] for x in item["entities list"]]
            item_entities_list = [item_matcher[x][0] for x in item["item entities list"]]

            possible_entities_list = [x for x in item_ent2mention]

            # sort lists
            entity_list = sort_entities_list(entity_list, entities_ordering)
            possible_entities_list = sort_entities_list(possible_entities_list, entities_ordering)
            item_entities_list = sort_entities_list(item_entities_list, entities_ordering)
            
            item["auto entities list"] = entity_list                     # list of GT entities in doc (converted to auto)
            item["auto item entities list"] = item_entities_list         # list of GT entitiies in item (converted to auto)
            item["entities list"] = flatten([auto2gt[x] for x in entity_list])                        # list of GT entities in doc
            item["item entities list"] = flatten([auto2gt_item[x] for x in item_entities_list])            # list of GT entitiies in item

            new_item_entities = sort_entities_pairs(new_item_entities, entities_ordering)
            item["entities"] = new_item_entities
            
            new_data.append(item)

    return {"name":file_name,"data":new_data}

def sort_entities_list(entities, reference):
    new_entities = []
    for x in reference:
        if x in entities:
            new_entities.append(x)
    return new_entities

def sort_entities_pairs(entities, reference):
    reference_scores = {x:len(reference)-reference.index(x) for x in reference}
    entities_scores = []
    for x in entities:
        score = 0
        if x[0]["auto_match"] in reference_scores:
            score += reference_scores[x[0]["auto_match"]]
        if  x[1]["auto_match"] in reference_scores:
            score += reference_scores[x[1]["auto_match"]]

        entities_scores.append(score)
    assert len(entities_scores) == len(entities)
    new_entities = [x for x, _ in sorted(zip(entities, entities_scores), key=lambda pair: pair[1], reverse=True)]
    return new_entities

def save_data(datas:list):
    for data in datas:
        name = data['name']
        data_json = data["data"]
        with open('../'+name+"_standard.json", 'w') as outfile:
            json.dump(data_json, outfile, indent=4) 

if __name__ == "__main__":    
    db = None
    with open('../ideologyKB.json', 'r') as file:
        db = json.load(file)     

    db_data = {x: db[x][0] for x in db}  

    train = process_data('train', db_data)
    print(len(train['data']))
    dev= process_data('valid', db_data)
    print(len(dev['data']))
    # special treatment on test
    test = process_data('test', db_data, True)
    print(len(test['data']))

    save_data([train])
    save_data([dev])
    save_data([test])
