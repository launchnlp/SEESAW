import json
import os
import numpy as np
from graph_util import *
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def read_data(file):
    data = []
    with open(os.path.join('..', file+'_standard.json'), 'r') as f:
        data =json.load(f)
    return data


def process_data(data, level:str):
    # level = ["article", "context", "text"]
    new_data = []
    count = 0
    look_up_ref = {}
    # triplet_count=(0,0,0)
    for item in tqdm(data):
        new_item = {}
        text = None
        if level == "article":
            text = item["document text"]
        elif level == "context":
            text = item["context text"]
        elif level == "text":
            text = item["text"]
        
        if text in look_up_ref:
            triples, triples_idx, coref_doc, cluster_mention_name = look_up_ref[text]
        else:
            triples, triples_idx, coref_doc, cluster_mention_name = sentence_process(text)
            look_up_ref[text] = (triples, triples_idx, coref_doc, cluster_mention_name)
            
        if triples == None:
            print (count)
            count+=1
            continue

        new_item["triples"] = triples
        new_item["triples_idx"] = triples_idx
        new_item['coref'] = cluster_mention_name
        new_data.append(new_item)
    return new_data

def save_data(name, data, level):
    data_json = data
    with open('../graph/'+name+"."+level+".json", 'w') as outfile:
        json.dump(data_json, outfile, indent=4, cls=NpEncoder) 


for x in ["train", "valid", "test"]:
    data = read_data(x)
    data = process_data(data, level="context")
    save_data(x, data, level="context")
    # data = process_data(data, level="article")
    # save_data(x, data, level="article")

