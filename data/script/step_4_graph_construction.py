from asyncio import new_event_loop
from audioop import add
import json
import os
from tkinter import E
import numpy as np
import time
# from graph_util import *
from tqdm import tqdm
import csv
import pickle 
import spacy
from wikipedia2vec import Wikipedia2Vec
from google.cloud import language_v1
import time
import itertools

MODEL_FILE = "../enwiki_20180420_500d.pkl"
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

def read_data(file, version):
    data = []
    with open(os.path.join('..', file+''+'_standard.json'), 'r') as f, open(os.path.join('../graph', file+'.'+version+'.json'), 'r') as f1, open(os.path.join('../cache', 'data.entities2mentions.'+file+'.json'), 'r') as f2:
        data =json.load(f)
        data_graph =json.load(f1)  
        data_entities =json.load(f2)
    assert len(data) == len(data_entities)
    assert len(data) == len(data_graph)
    return data, data_graph, data_entities


def update_item(item, offset):
    corefs = item['coref']
    triples_idx = item["triples_idx"]
    for coref in corefs:
        for span in corefs[coref]:
            span[0] += offset
            span[1] += offset
    for triple in triples_idx:
        for span in triple:
            span[0] += offset
            span[1] += offset
    return item

def process_data_fast(data, data_graph, data_entities, external_enabled=False, enhance=True, graph_completeness="article", ents_completeness="doc_ents", prune_enabled=True, data_split=None):
    # if enhance = True, then we will link mention node to entity node.
    collections = []
    counter = 0
    for item, item_graph, item_entities in tqdm(zip(data, data_graph, data_entities)):
        text = item['text']
        context_text = item['context text']
        document_text = item['document text']
        text_start_char = document_text.index(text)
        text_start = len(document_text[0:text_start_char].split())
        text_end = text_start + len(text.split())
        context_start_char = document_text.index(context_text)
        context_start = len(document_text[0:context_start_char].split())
        context_end = context_start + len(context_text.split())
        OFFSET = context_start if graph_completeness=="context" else 0
        entity_ordering = item["entities ordering"]

        # update local/relative index to global index
        item_graph = update_item(item_graph, OFFSET)

        triples = item_graph["triples"]
        triples_idx = item_graph["triples_idx"]
        corefs = item_graph["coref"]
        if data_split == "test":
            doc_entities = item["auto entities list"]
            item_entities_list = item["auto item entities list"]
        else:
            doc_entities = item["entities list"]
            item_entities_list = item["item entities list"]

        node_collection = []
        edge_collection = []
        span2node = {}


        if ents_completeness != "no_ents":
            for entity in item_entities:
                spans, wiki_link = item_entities[entity][0], item_entities[entity][1]
                node_id = len(node_collection)

                span_collection = [x[2] for x in spans]
                if ents_completeness == "context_ents":
                    if any(element[0]>=context_start and element[1]<= context_end for element in span_collection):
                        pass
                    else:
                        continue
                node_collection.append({"node_id":node_id, "name":entity, "spans":span_collection, "wiki":wiki_link, "type":"entity"})
                for element in node_collection[node_id]["spans"]:
                    span2node[str(element)] = node_id
            assert len(node_collection)!=0

        for entity in corefs:
            new_node = True
            for span in corefs[entity]:    
                # 
                if str(span) in span2node:
                    new_node = False
                    node_to_be_updated = node_collection[span2node[str(span)]]
                    for element in corefs[entity]:
                        span2node[str(element)] = span2node[str(span)]
                        node_to_be_updated["spans"].append(element)
                    break
            if new_node:
                node_id = len(node_collection)
                node_collection.append({"node_id":node_id, "name":entity, "spans":corefs[entity], "wiki": "", "type":"mention"})
                for element in corefs[entity]:
                    span2node[str(element)] = node_id

                if enhance: 
                    for span in node_collection[node_id]["spans"]:
                        for node in node_collection:
                            if node["type"] == "entity":
                                if any(element[0]>=span[0] and element[1]<= span[1] for element in node["spans"]):
                                    if [node["node_id"], span2node[str(span)]] not in edge_collection and node["node_id"] != span2node[str(span)]:
                                        # connect entity to mention
                                        edge_collection.append([node["node_id"], span2node[str(span)]])

        node_collection_copy = node_collection.copy()
        if external_enabled == True or external_enabled=="wiki":
            for x in node_collection_copy:
                node_id = x["node_id"]
                canonical = x["name"]
                wiki = x["wiki"]

                # add wiki node
                if wiki != "" and wiki2vec.get_entity(canonical) != None:
                    vector = wiki2vec.get_entity_vector(canonical)
                    wiki_node_id = len(node_collection)
                    node_collection.append({"node_id":wiki_node_id, "name":canonical, "type":"wiki", "vector":vector})
                    span2node[canonical+"_entity"] = wiki_node_id
                    # connect entity to wiki entity
                    edge_collection.append([node_id, span2node[canonical+"_entity"]])


        assert len(triples) == len(triples_idx)
        for triple, idx in zip(triples, triples_idx):
            # remove spans that are too long
            if idx[0][1]-idx[0][0] >15 or idx[2][1]-idx[2][0] >15 or idx[1][1]-idx[1][0] > 5:
                continue

            for i, (name, span) in enumerate(zip(triple, idx)):
                if i!=1:
                    if str(span) not in span2node:
                        node_id = len(node_collection)
                        node_collection.append({"node_id":node_id, "name":name, "spans":[span], "wiki": "", "type":"mention"})
                        span2node[str(span)] = node_id
                        if enhance:
                            for node in node_collection:
                                if node["type"] == "entity":
                                    if any(element[0]>=span[0] and element[1]<= span[1] for element in node["spans"]):
                                        if [node["node_id"], span2node[str(span)]] not in edge_collection and node["node_id"] != span2node[str(span)]:
                                            edge_collection.append([node["node_id"], span2node[str(span)]]) 
                    else:
                        node_to_be_updated = node_collection[span2node[str(span)]]  
                        node_to_be_updated["spans"].append(span)      
                                      
                else:
                    # for predicate, we de-dupilicate using both the name and the span
                    # it's possible that some predicates are already added as "entity", e.g., "block"
                    if name not in span2node and str(span) not in span2node:
                        # and str(span) not in span2node
                        node_id = len(node_collection)
                        node_collection.append({"node_id":node_id, "name":name, "spans":[span], "wiki": "", "type":"predicate"})
                        span2node[name] = node_id 
                        span2node[str(span)] = node_id
                    else:
                        identifier = span2node[name] if name in span2node else span2node[str(span)]
                        node_to_be_updated = node_collection[identifier]  
                        span2node[name] = node_to_be_updated["node_id"]
                        span2node[str(span)] = node_to_be_updated["node_id"]
                        node_to_be_updated["spans"].append(span)                
            
            # Add edges between predicate and entity/mention nodes
            if [span2node[str(idx[0])],span2node[str(triple[1])]] not in edge_collection:
                edge_collection.append([span2node[str(idx[0])],span2node[str(triple[1])]])
            if [span2node[str(triple[1])],span2node[str(idx[2])]] not in edge_collection:
                edge_collection.append([span2node[str(triple[1])],span2node[str(idx[2])]])
                      
        # enhance node representations with whether in_context and whether in_text
        for node in node_collection:
            if node["type"] in ["wiki", "social"]:
                # this is a wiki (i.e., external) entity
                continue
            k = node["spans"]
            k.sort()
            node["spans"] = list(k for k,_ in itertools.groupby(k)) 

            if node["type"] == "entity" or node["type"] == "mention":
                if any(span[0]>=context_start and span[1]<=text_start for span in node["spans"]):
                    node["in_left_context"] = True
                else:
                    node["in_left_context"] = False
                if any(span[0]>=text_start and span[1]<=text_end for span in node["spans"]):
                    node["in_text"] = True
                else:
                    node["in_text"] = False
                if any(span[0]>=text_end and span[1]<=context_end for span in node["spans"]):
                    node["in_right_context"] = True
                else:
                    node["in_right_context"] = False
            elif node["type"] == "predicate":
                if node["name"].lower().strip() in " ".join(context_text.split()[:text_start]).lower().strip():
                    node["in_left_context"] = True
                else:
                    node["in_left_context"] = False
                if node["name"].lower().strip() in text.lower().strip():
                    node["in_text"] = True
                else:
                    node["in_text"] = False
                if node["name"].lower().strip() in " ".join(context_text.split()[text_end:]).lower().strip():
                    node["in_right_context"] = True
                else:
                    node["in_right_context"] = False       

        if prune_enabled:
            # prune all the nodes that are not connected to any other nodes
            new_available_node_ids = []
            new_node_collection = []

            for x in edge_collection:
                if x[0] == x[1]:
                    print("shoot")
                new_available_node_ids.append(x[0])
                new_available_node_ids.append(x[1])
            new_available_node_ids = set(new_available_node_ids)

            for node in node_collection:
                if node["node_id"] in new_available_node_ids:
                    new_node_collection.append(node)  
            node_collection = new_node_collection  

        collections.append({"nodes":node_collection, "edges":edge_collection, "entities":item_entities_list, "doc_entities": doc_entities, "entity_ordering": entity_ordering})
    return collections

for x in ["train", "valid", "test"]:
    for y,z in zip(["article", "context"],["doc_ents", "context_ents"]):      
            for external in [True, False]:
                for prune in [False]:
                    data, data_graph, data_entities = read_data(x, y)
                    collections = process_data_fast(data, data_graph, data_entities, external, enhance=True, graph_completeness=y, ents_completeness=z, prune_enabled=prune, data_split = x)
                    if external==True:
                        if prune:
                            pickle.dump(collections, open("../graph/"+x+"."+y+"."+z+"_external_prune.pkl", "wb" ) )
                        else:
                            pickle.dump(collections, open("../graph/"+x+"."+y+"."+z+"_external.pkl", "wb" ) )
                    elif external ==False:
                        if prune:
                            pickle.dump(collections, open("../graph/"+x+"."+y+"."+z+"_prune.pkl", "wb" ) )
                        else:
                            pickle.dump(collections, open("../graph/"+x+"."+y+"."+z+".pkl", "wb" ) )