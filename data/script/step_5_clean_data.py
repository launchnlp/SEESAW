import pickle
import json


for file in ["train", "valid", "test"]:
    with open('../'+file+'_standard.json', 'r') as json_file:    
        new_data = []
        data = json.load(json_file)

        for item in data:
            new_item = {}
            new_item["text"] = item["text"]
            new_item["context text"] = item["context text"]
            new_item["document text"] = item["document text"]
            new_item["entities"] = item["entities"]
            new_item["unique_id"] = item["unique_id"]
            new_data.append(new_item)

    with open('../'+file+'.json', 'w') as outfile:
        json.dump(new_data, outfile, indent=4)   