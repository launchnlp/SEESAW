from curses import raw
from os import listdir
import os
from os.path import isfile, join  
import glob
import json
import csv
import random
from queue import PriorityQueue
import numpy as np 
import requests
import spacy
import time
import ast

from google.cloud import language_v1
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from data_conversion import *

nlp = English()
nlp = spacy.load("en_core_web_sm")

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt


random.seed(10)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "PATH_TO_YOUR_GOOGLE_CLOUD_CREDETIAL"
client = language_v1.LanguageServiceClient()


def translate_to_en_wiki(entity_name, from_lang):
    if from_lang == "en":
        entity_name = " ".join(entity_name.split("_"))
        return entity_name, "https://en.wikipedia.org/wiki/{}".format("_".join(entity_name.split()))
    _S="https://{}.wikipedia.org/w/api.php?action=query&format=json&prop=langlinks&titles={}&llprop=autonym|langname&lllimit=500".format(from_lang, entity_name)
    req = requests.get(_S)
    json_string = json.loads(req.text)
    wiki_id = list(json_string["query"]["pages"].keys())[0]
    langs = json_string["query"]["pages"][wiki_id]['langlinks']
    new_entity = None
    new_url = None
    for x in langs:
        if x['langname'] == 'Englisch' or x['lang'] == "en":
            new_entity =  x['*'] 
            new_url = "https://en.wikipedia.org/wiki/{}".format("_".join(new_entity.split()))
            new_entity = " ".join(x['*'].split("_"))
    return new_entity, new_url

def analyze_entity_sentiment(text_content, client):
    """
    Analyzing Entity Sentiment in a String

    Args:
      text_content The text content to analyze
    """
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}
    encoding_type = language_v1.EncodingType.UTF16
    response = client.analyze_entities(request = {'document': document, 'encoding_type': encoding_type})
    entity_set = []
    for entity in response.entities:
        entity_name = entity.name
        meta_data = {}
        mentions = []
        for metadata_name, metadata_value in entity.metadata.items():
            meta_data[metadata_name] = metadata_value
            if metadata_name == "wikipedia_url":
                tmp_entity_name, from_lang = metadata_value.split('/wiki/')[-1], metadata_value.split('//')[1].split(".")[0]
                try:
                    tmp_entity_name, eng_url = translate_to_en_wiki(tmp_entity_name, from_lang)
                    if tmp_entity_name != None:
                        entity_name = tmp_entity_name.strip() if "(" not in tmp_entity_name else tmp_entity_name[0:tmp_entity_name.index("(")].strip()
                        meta_data[metadata_name] = eng_url
                except:
                    print(entity_name)
                    print("Didn't find English wiki page")
        for mention in entity.mentions:
            mention_content = mention.text.content
            mention_char_start = mention.text.begin_offset
            mention_start = len(text_content[:mention_char_start].split())
            mention_end = mention_start + len(mention_content.split())
            if text_content.split()[mention_start:mention_end] == mention_content:
                print(mention_content, text_content.split()[mention_start:mention_end])
            mention_type = language_v1.EntityMention.Type(mention.type_).name
            mentions.append((mention_content, mention_type,(mention_start, mention_end)))
        entity_set.append((entity_name, language_v1.Entity.Type(entity.type_).name, meta_data, mentions))
    return entity_set

def process_data(file):
    data = []
    article_ids = []
    with open("../SEESAW_data."+file+".json", 'r') as json_file:
        article = json.load(json_file)
        for item in tqdm(article):
            article_id = item["unique_id"].split("/")[0]
            if article_id not in article_ids:
                article_ids.append(article_id)
            else:
                data.append({'mentions': union_list, 'linked_entities': list_w_wiki, "entities_mentions": entities_mentions})
                continue

            document = item["document text"]
            doc = nlp(document)
            tokens = list(doc)
            doc_tokens = " ".join([str(x) for x in tokens])

            entities = [(" ".join([str(x) for x in list(x)]), x.label_, (x.start, x.end)) for x in list(doc.ents)]
            noun_chunks = [(" ".join([str(x) for x in list(x)]), "",  (x.start, x.end)) for x in list(doc.noun_chunks)]
            linked_entities_analysis = analyze_entity_sentiment(doc_tokens, client)
            linked_entities = [(x[0], entity[1], x[2]) for entity in linked_entities_analysis for x in entity[3]]

            union_list = list(set(entities) | set(noun_chunks) | set(linked_entities))
            list_w_wiki = [(x[0], x[2]['wikipedia_url']) for x in linked_entities_analysis if 'wikipedia_url' in x[2]]
            entities_mentions = dict()
            for x in linked_entities_analysis:
                wiki = ""
                if 'wikipedia_url' in x[2]:
                    wiki = x[2]['wikipedia_url']
                spans = [item for item in x[3] if item[0] == " ".join(doc_tokens.split()[item[2][0]:item[2][1]])]
                # handle the situation where we end up with an empty list of spans
                if len(spans) == 0:
                    continue
                entities_mentions[x[0]] = (x[1], wiki, spans)

            # filter out unmatched items (i.e., doc.split()[start index:end index]] != mention text)
            union_list = [item for item in union_list if item[0] == " ".join(doc_tokens.split()[item[2][0]:item[2][1]])]

            for item in union_list:
                if item[0] == " ".join(doc_tokens.split()[item[2][0]:item[2][1]]):
                    pass
                else:
                    print(item[0])
                    print(" ".join(doc_tokens.split()[item[2][0]:item[2][1]]))

            for entity in entities_mentions:
                items = entities_mentions[entity][2]

                for item in items:
                    if item[0] == " ".join(doc_tokens.split()[item[2][0]:item[2][1]]):
                        pass
                    else:
                        print(item[0])
                        print(" ".join(doc_tokens.split()[item[2][0]:item[2][1]]))
                        print(item[2])
                        print(doc_tokens)
            
            data.append({'mentions': union_list, 'linked_entities': list_w_wiki, "entities_mentions": entities_mentions})
    return data



for file in ["train", "valid", "test"]:
    with open('../cache/data.raw.entities.'+file+'.json', 'w') as outfile:
        data = process_data(file)
        json.dump(data, outfile, indent=4)  

    data, data_entities = read_data_conversion(file)
    entity_list_collection, possible_entities_collection  = process_entities(data, data_entities)

    with open('../cache/data.entities2mentions.'+file+'.json', 'w') as outfile:
        json.dump(possible_entities_collection, outfile, indent=4)  
    with open('../cache/data.entities_automatch.'+file+'.json', 'w') as outfile:
        json.dump(entity_list_collection, outfile, indent=4)    