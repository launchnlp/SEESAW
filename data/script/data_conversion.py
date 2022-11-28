from hashlib import new
import json
import os
from tabnanny import check
from turtle import pos
from click import launch
import numpy as np
from spacy.lang.en import English
import spacy
from tqdm import tqdm
import requests
import string
from nltk.corpus import stopwords
import numpy as np
from scipy import stats
from difflib import get_close_matches
import ast
import statistics
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def isplural(word):
    lemma = wnl.lemmatize(word, 'n')
    plural = True if word is not lemma else False
    return plural, lemma



nlp = English()
nlp = spacy.load("en_core_web_sm")


punctuations = set(string.punctuation)
stop_words1 = set(stopwords.words('english'))
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stop_words2 = set(stopwords_list.decode().splitlines())

removable = punctuations.union(stop_words1).union(stop_words2)
removable = removable.union(set([x for x in removable]))

removable = set([x.lower() for x in removable])

month_list = set(["January", "Jan.", "February", "Feb.", "March", "Mar.", "April", "Apr.", "May", "May.", "June", "Jun.", "July", "Jul.", "August", "Aug.", "September", "Sep.", "October", "Oct.", "November", "Nov.", "December", "Dec."])


def fix_linking(collections, linked_entities):
    # heuristic rules for fixing entity extraction and linking on 1) house/white house/senate; 2) undocumented immigrants; 3) republican/democrats; 4) us-mexico border; 5) lawmaker
    new_collections = dict()
    old_collections = dict()

    for entity in collections:
        # section 1: house/white house/senate
        if entity.lower() == "the house":
            new_collections["United States House of Representatives"] = collections[entity]
            linked_entities["United States House of Representatives"]="https://en.wikipedia.org/wiki/United_States_House_of_Representatives"
        elif entity.lower() == "the white house":   
            new_collections["White House"] = collections[entity]
            linked_entities["White House"]="https://en.wikipedia.org/wiki/White_House"
        elif entity.lower() == "senate":
            new_collections["United States Senate"] = collections[entity]
            linked_entities["United States Senate"]="https://en.wikipedia.org/wiki/United_States_Senate"
        elif entity.lower() == "house":
            mentions = collections[entity]
            WH_list = []
            House_list = []
            for mention in mentions:
                if mention[0].lower() in ["white house", "the white house"]:
                    WH_list.append(mention)
                else:
                    House_list.append(mention)
            if len(WH_list) > 0:
                new_collections["White House"] = WH_list
                linked_entities["White House"]="https://en.wikipedia.org/wiki/White_House"
            if len(House_list) > 0:
                new_collections["United States House of Representatives"] = House_list
                linked_entities["United States House of Representatives"]= "https://en.wikipedia.org/wiki/United_States_House_of_Representatives"
        elif entity.lower() == "white house":
            mentions = collections[entity]
            House_list = []
            WH_list = []
            for mention in mentions:
                if mention[0].lower() in ["house", "the house"]:
                    House_list.append(mention)
                else:
                    WH_list.append(mention)
            if len(WH_list) > 0:
                new_collections["White House"] = WH_list
                linked_entities["White House"]="https://en.wikipedia.org/wiki/White_House"
            if len(House_list) > 0:
                new_collections["United States House of Representatives"] = House_list
                linked_entities["United States House of Representatives"]= "https://en.wikipedia.org/wiki/United_States_House_of_Representatives"
        # section 2: undocumented immigrants
        elif entity.lower() == "immigrant": 
            mentions = collections[entity]
            find = False          
            for mention in mentions:
                if mention[0].lower() in ["undocumented immigrants", "undocumented immigration", "illegal immigrants", "illegal immigration"]:
                    new_collections["undocumented immigrants"] = collections[entity]
                    linked_entities["undocumented immigrants"] = "https://en.wikipedia.org/wiki/Illegal_immigration"
                    find = True
                    break
            if not find:
                new_collections[entity] = collections[entity]
        # section 3: republican/democrats
        elif entity.lower() in ["republican", "republic", "the republic", "republicans"]:
            mentions = collections[entity]
            new_collections["Republican Party"] = collections[entity]
            linked_entities["Republican Party"] = "https://en.wikipedia.org/wiki/Republican_Party_(United_States)"
        elif entity.lower() in ["democrat", "democratic", "democrats"]:
            mentions = collections[entity]
            new_collections["Democratic_Party"] = collections[entity]
            linked_entities["Democratic_Party"] = "https://en.wikipedia.org/wiki/Democratic_Party_(United_States)"
        # section 4: us-mexico border and border wall
        elif entity.lower().split()[-1] == "border":
            mentions = collections[entity]
            new_collections["US-Mexico border"] = collections[entity]
            linked_entities["US-Mexico border"] = "https://en.wikipedia.org/wiki/Mexico%E2%80%93United_States_border"
        elif entity.lower() in ["border wall", "the border wall"]:
            mentions = collections[entity]
            new_collections["US-Mexico border wall"] = collections[entity]
            linked_entities["US-Mexico border wall"] = "https://en.wikipedia.org/wiki/Mexico%E2%80%93United_States_barrier"
        # section 5: lawmaker
        elif "lawmaker" in entity.lower() and entity.lower() != "lawmaker":
            lawmaker_list = []
            others_list = []
            for mention in collections[entity]:
                if mention[0].lower() == "lawmaker" or mention[0].lower() == "lawmakers":
                    lawmaker_list.append(mention)
                else:
                    others_list.append(mention)
            if len(lawmaker_list) > 0:
                new_collections["lawmaker"] = lawmaker_list  
            if len(others_list) > 0:
                new_collections[entity] = others_list         
        else:
            old_collections[entity] = collections[entity]

    # merge collections
    for entity in new_collections:
        if entity in old_collections:
            old_collections[entity] = old_collections[entity] + new_collections[entity]
        else:
            old_collections[entity] = new_collections[entity]

    return old_collections, linked_entities

def read_data_conversion(file):
    with open("../SEESAW_data."+file+".json", 'r') as f:
        data =json.load(f)
    with open("../cache/data.raw.entities."+file+'.json', 'r') as f:
        data_entities =json.load(f)    
    return data, data_entities

def extract_entity_mention_pair(item_entity):
    linked_entities = {x[0]: x[1]for x in item_entity['linked_entities']}
    entity2mention = dict()
    # entity2mention: key: canonical entity, value: list of entity mentions (mention, type, [start, end])
    for x in item_entity['entities_mentions']:
        y = item_entity['entities_mentions'][x]
        if y[0] != "NUMBER" and y[0] != "DATE" and y[0] != "TIME" and y[0] != "QUANTITY" and y[0]!="PRICE" and y[0]!="MONEY" and y[0]!="CARDINAL" and x not in removable:
            entity2mention[x] = [(t[0], y[0], t[2]) for t in y[2] if t[1] != "TYPE_UNKNOWN"]
            if len(entity2mention[x]) == 0:
                entity2mention.pop(x)

    # 
    initial_mention_span2entity = dict()
    # x[0] is the mention, x[1] is the type, x[2] is the span
    for x in entity2mention:
        for y in entity2mention[x]:
            initial_mention_span2entity[str(y[2])] = x
    # 
    entity_mentions = []
    for x in item_entity["mentions"]:
        if x[1] != "NUMBER" and x[1] != "DATE" and x[1] != "TIME" and x[1] != "QUANTITY" and x[1]!="PRICE" and x[1]!="MONEY"  and x[1]!="CARDINAL":
            item = x[0].lower()
            entity_mentions.append(item)
    entity_mentions = set(entity_mentions)
    entity_mentions = entity_mentions.difference(removable).difference(month_list)
    
# 
    # Expansion step 1: if candidate span is a subspan or a superspan of an entity mention (i.e., the mentions that have been mapped to an entity) or contains a portion of an entity mention. USING SPAN to match
    possible_mentions_step1 = [x for x in item_entity["mentions"] if str(x[2]) not in initial_mention_span2entity and x[0].lower() in entity_mentions]

    mention_span2entity_step1 = dict()
    for x in possible_mentions_step1:
        candiate_span = x[2]
        for y in initial_mention_span2entity:
            span = ast.literal_eval(y)
            if (candiate_span[0] >= span[0] and candiate_span[1] <= span[1]) or  (candiate_span[0] <= span[0] and candiate_span[1] >= span[1]) or (candiate_span[0] <= span[0] and candiate_span[1] <= span[1] and candiate_span[1] > span[0]) or (candiate_span[0] >= span[0] and candiate_span[0] < span[1] and candiate_span[1] >= span[1]):
                if str(candiate_span) not in mention_span2entity_step1:
                    mention_span2entity_step1[str(candiate_span)] = [(x[0], x[1], candiate_span, span, initial_mention_span2entity[y])]
                else:
                    mention_span2entity_step1[str(candiate_span)].append((x[0], x[1], candiate_span, span, initial_mention_span2entity[y]))
    mention_span2entity_second = {}


    for x in mention_span2entity_step1:
        if len(set(t[-1] for t in mention_span2entity_step1[x])) == 1:
            mention_span2entity_second[x] = mention_span2entity_step1[x][0][-1]
            entity2mention[mention_span2entity_step1[x][0][-1]].append((mention_span2entity_step1[x][0][0], mention_span2entity_step1[x][0][1], mention_span2entity_step1[x][0][2]))
        else:
            mention = mention_span2entity_step1[x][0][0]
            type_ = mention_span2entity_step1[x][0][1]
            doc = nlp(mention)
            head = [x["id"] for x in doc.to_json()["tokens"] if x["dep"] == "ROOT"][0]
            head_id = mention_span2entity_step1[x][0][2][0] + head
            candidates_linked = []
            candidates_head = []
            candidates_head_linked = []
            for candidate in mention_span2entity_step1[x]:
                canonical_form = candidate[-1]
                lucky_check = 0
                if canonical_form in linked_entities:
                    candidates_linked.append(candidate)
                    lucky_check += 1
                if head_id >= candidate[3][0] and head_id < candidate[3][1]:
                    candidates_head.append(candidate)
                    lucky_check += 1
                if lucky_check == 2:
                    candidates_head_linked.append(candidate)
                    # 
            Finished = False

            if len(candidates_head_linked) == 1:
                entity = candidates_head_linked[0][-1]
                mention_span2entity_second[x] = entity
                entity2mention[entity].append((mention, type_, candidate[2]))
                Finished = True
            elif len(candidates_head_linked) > 1:
                stat = stats.mode([x[-1] for x in candidates_head_linked])
                if stat[1][0]>1:
                    entity = stat[0][0]
                    mention_span2entity_second[x] = entity
                    entity2mention[entity].append((mention, type_, candidate[2]))
                    Finished = True
            if not Finished:
                if len(candidates_linked) == 1:
                    entity = candidates_linked[0][-1]
                    mention_span2entity_second[x] = entity
                    entity2mention[entity].append((mention, type_, candidate[2]))
                    Finished = True
                elif len(candidates_linked) > 1:
                    stat = stats.mode([x[-1] for x in candidates_linked])
                    if stat[1][0]>1:
                        entity = stat[0][0]
                        mention_span2entity_second[x] = entity
                        entity2mention[entity].append((mention, type_, candidate[2]))
                        Finished = True
            if not Finished:
                if len(candidates_head) == 1:
                    entity = candidates_head[0][-1]
                    mention_span2entity_second[x] = entity
                    entity2mention[entity].append((mention, type_, candidate[2]))
                    Finished = True
                elif len(candidates_head) > 1:
                    stat = stats.mode([x[-1] for x in candidates_head])
                    if stat[1][0]>1:
                        entity = stat[0][0]
                        mention_span2entity_second[x] = entity
                        entity2mention[entity].append((mention, type_, candidate[2]))
                        Finished = True 
            if not Finished:
                mention_span2entity_second[x] = mention
                entity2mention[mention] = [(mention, mention_span2entity_step1[x][0][1], mention_span2entity_step1[x][0][2])]                         
# 
    # Expansion step 2: if a mention is part of an entity mention or an entity mention is part of a mention, using MENTION to match
    possible_mentions_step2 = [x for x in item_entity["mentions"] if str(x[2]) not in initial_mention_span2entity and x[0].lower() in entity_mentions and str(x[2]) not in mention_span2entity_second]

    initial_mention2entity= dict()
    mention2entity_first = dict()
    for x in entity2mention:
        for y in entity2mention[x]:
            if y[0] not in initial_mention2entity:
                initial_mention2entity[y[0]] = set([x])
            else:
                initial_mention2entity[y[0]].add(x)
# 
    possible_mentions_step2_look_up = dict()
    for x in possible_mentions_step2:
        mention = x[0]
        type_ = x[1]
        span = x[2]
        if mention not in possible_mentions_step2_look_up:
            possible_mentions_step2_look_up[mention] = [(type_, span)]
        else:
            possible_mentions_step2_look_up[mention].append((type_, span))
        for y in initial_mention2entity:
            if mention.lower() ==y.lower():
                mention2entity_first[mention] = list(initial_mention2entity[y])
            elif mention.lower() in y.lower() or y.lower() in mention.lower():
                if mention not in mention2entity_first:
                    mention2entity_first[mention] = list(initial_mention2entity[y])
                else:
                    mention2entity_first[mention].extend(list(initial_mention2entity[y]))
# 
    mention2entity_second = dict()
    for x in mention2entity_first:
        if len(mention2entity_first[x]) == 1:
            mention2entity_second[x] = mention2entity_first[x][0]
        else:
            # t is a entity
            tmp =  [t for t in mention2entity_first[x] if t in linked_entities]
            if len(set(tmp)) == 1:
                mention2entity_second[x] = tmp[0]
            elif len(set(tmp)) > 1:
                mention2entity_second[x] = x
            else:
                tmp = [t for t in mention2entity_first[x] if t in x or x in t]
                if len(set(tmp)) == 1:
                    mention2entity_second[x] = tmp[0]
                elif len(set(tmp)) > 1:
                    mention2entity_second[x] = tmp[np.argmax([len(t) for t in tmp])]
                else:
                    mention2entity_second[x] = x
        if mention2entity_second[x] in entity2mention:
            entity2mention[mention2entity_second[x]].extend([(x, p[0], p[1]) for p in possible_mentions_step2_look_up[x]])
        else:
            entity2mention[mention2entity_second[x]] = [(x, p[0], p[1]) for p in possible_mentions_step2_look_up[x]]
    
    # These unmatched mentions are usually not important
    possible_mentions_step3 = [x for x in item_entity["mentions"] if str(x[2]) not in initial_mention_span2entity and x[0].lower() in entity_mentions and str(x[2]) not in mention_span2entity_second and x[0] not in mention2entity_second]
    
    omitted_count = len(possible_mentions_step3)

    # further reduce entity2mention size by merging (singular and plural), (upper and lowercases) 
    # I lowercase all entities' canonical names here, and covert all plural names to singular names (exceptions: if an entity has a linked page, i.e., already in canonical form, don't remove trailing 's')
    # Keep two copies: one with lowercase, one with original form
    
    # entity: {link:, canonical:, singular_lower:}
    entityCollections = dict()  
    orig2singularLower = dict() 
    for x in entity2mention:
        
        link = linked_entities[x] if x in linked_entities else ""
        canonical = x if link != "" else ""
        singularLower = None
        if len(x.strip().split()) == 1:
            singularLower = wnl.lemmatize(x.strip().lower(), 'n')
        else:
            singularLower = " ".join(x.split()[:-1] + [wnl.lemmatize(x.split()[-1].strip().lower(), 'n')]).strip().lower() 
                    
        entityCollections[x] = {"link": link, "canonical": canonical, "singularLower": singularLower}
        orig2singularLower[x] = singularLower

    uniqueEntityCollection = set([entityCollections[x]["singularLower"] for x in entityCollections])
    singularLower2standard = dict()
    for x in uniqueEntityCollection:
        for _, y in entityCollections.items():
            if y["singularLower"] == x:
                if y["singularLower"] not in singularLower2standard:
                    singularLower2standard[x] = y["singularLower"] if y["canonical"] == "" else y["canonical"]
                else:
                    singularLower2standard[x] = y["canonical"] if y["canonical"]!= "" else singularLower2standard[x]
                
    entity2mention_new = dict()

    for x in entity2mention:      
        singularLower = orig2singularLower[x.strip()]
        standard = singularLower2standard[singularLower]

        if standard not in  entity2mention_new:
            entity2mention_new[standard] = entity2mention[x]
        else:
            entity2mention_new[standard].extend(entity2mention[x])

            # print(singular)
            # print(entity2mention_new[singular])
    entity2mention_standard = entity2mention_new
    entity2mention_standard, linked_entities = fix_linking(entity2mention_standard, linked_entities)
    entity2mention = {x.lower():y for x,y in entity2mention_standard.items()}

    mention2entity = dict()
    for entity in entity2mention:
        for y in entity2mention[entity]:
            if y[0] not in mention2entity:
                mention2entity[y[0].lower()] = set([entity])
            else:
                mention2entity[y[0].lower()].add(entity)
    linked_entities = {k.lower():v for k,v in linked_entities.items()}

    return  mention2entity, entity2mention, linked_entities, omitted_count, entity2mention_standard


def process_entities(data, data_entities):
    articles_collection =  []
    count_match = 0
    count_partial = 0
    count_no_match = 0
    entity_list_collection = []  # map mentions to entities
    possible_entities_collection = []
    count_entities = []
    count_mentions = []
    counter = 0
    omitted_count  = 0
    for item , item_entity in tqdm(zip(data,data_entities)):
        unique_id = item["unique_id"].split("/")[0]

        if unique_id not in  articles_collection:
            articles_collection.append(unique_id)
        else:
            entity_list_collection.append(new_entity_list_ref)  
            possible_entities_collection.append(entity2mention_standard)
            continue

        # lowercase the following two
        annotator_entities_standard = item["entities list"]
        annotator_entities = [x.lower() for x in annotator_entities_standard]
        automatic_mentions = [x[0].lower() for x in item_entity['mentions']]   # actually, it's not really an entity list

        counter += 1

        mention2entity, entity2mention, linked_entities, omitted_count_one_article, entity2mention_standard = extract_entity_mention_pair(item_entity)
        omitted_count += omitted_count_one_article
        count_entities.append(len(entity2mention))
        count_mentions.append(len(mention2entity))
        automatic_mentions = [x.lower() for x in mention2entity]  
        automatic_entities = [x.lower() for x in entity2mention] 
        # 
        match_list = {}
        new_entity_list ={}
        for x in annotator_entities:
            # find the singularLower version of the entity
            singularLower = None
            if len(x.strip().split()) == 1:
                singularLower = wnl.lemmatize(x.strip().lower(), 'n')
            else:
                singularLower = " ".join(x.split()[:-1] + [wnl.lemmatize(x.split()[-1].strip().lower(), 'n')]).strip().lower() 

            checker =  True
            similar_found_set= set()
            for y in automatic_entities:
                if x == y or singularLower==y.lower():
                    checker = False
                    match_list[x] = y
                    count_match += 1
                    break
                # if x in y or y in x:
                #     checker = False
                #     similar_found_set.add(y)
            for y in automatic_mentions:
                if x == y:            
                    checker = False
                    similar_found_set.add(y)
                    break
                if x in y or y in x:
                    for t in x.split():
                        if t in y.split():
                            checker =False
                            break
                    if not checker:
                        similar_found_set.add(y)                
            if x not in match_list and len(similar_found_set)>0:
                match_list[x] = similar_found_set
                count_partial += 1
            if checker:
                count_no_match += 1
                # print("{}: not found".format(x))
                # print(x in  item["document text"])
                match_list[x] = ""
            # 
            # match_List: annotator mention => entity / {mention(s)}

        for ann in match_list:
            # Mexico – United States barrier
            mention = match_list[ann]
            if mention != "":
                if isinstance(mention, str):
                    # though it's called "mention" here, it's actually an entity
                    new_entity_list[ann] = mention 
                else:
                    tmp_array = [x for x in mention]
                    tmp_array_entities = []
                    for x in mention:
                        tmp_array_entities.extend(list(mention2entity[x]))
                    doc = nlp(ann)
                    head = [x["id"] for x in doc.to_json()["tokens"] if x["dep"] == "ROOT"][0]
                    tmp_array_entities_linked = [x for x in tmp_array_entities if x in linked_entities]
                    tmp_array_entities_head = [x for x in tmp_array_entities if str(list(doc)[head]) in x]
                    tmp_array_head = [x for x in tmp_array if str(list(doc)[head]) in x]
                    # 
                    # if ann == "American Priority":
                    #     print(tmp_array_entities)
                    #     print(tmp_array_entities_linked)
                    #     print(tmp_array_entities_head)
                    lucky_check = [x.lower() for x in tmp_array_entities]
                    if ann.lower() in lucky_check:
                        new_entity_list[ann] = tmp_array_entities[lucky_check.index(ann.lower())]
                    else:
                        output = statistics.multimode(tmp_array_entities)
                        if len(output)==1:
                            new_entity_list[ann] = output[0]
                        else:
                            output = statistics.multimode([x for x in output if x in tmp_array_entities_linked])
                            if len(output)==1:
                                new_entity_list[ann] = output[0]
                            else:
                                output = statistics.multimode([x for x in output if x in tmp_array_entities_head])
                                if len(output)==1:
                                    new_entity_list[ann] = output[0]
                                else:
                                    mentions = get_close_matches(ann, tmp_array,cutoff=0.05)
                                    entities_tmp = list(mention2entity[mentions[0]])
                                    if len(entities_tmp)==1:
                                        new_entity_list[ann] = entities_tmp[0]
                                    else:
                                        entities_tmp = get_close_matches(ann, entities_tmp, cutoff=0.01)
                                        if len(entities_tmp) >0:
                                            new_entity_list[ann] = entities_tmp[0]
            else:
                # ad-hoc solution 
                tmp = get_close_matches(ann, list(entity2mention.keys()), cutoff=0.01)[0]
                if len(tmp) > 0:
                    new_entity_list[ann] = tmp
                else:
                    new_entity_list[ann] = ""

        # enhance new_entity_list (i.e., entity_convertor) with corresponding wiki link, if applicable
        for x in new_entity_list:
            wiki = ""
            if new_entity_list[x] in linked_entities:
                wiki = linked_entities[new_entity_list[x]]
            new_entity_list[x] = (new_entity_list[x], wiki)

        # enhance entity2mention and entity2mention_standard with corresponding wiki link
        
        # print(len(entity2mention))
        # print(len(entity2mention_standard))
        assert len(entity2mention_standard) == len(entity2mention)
        for x, y  in zip(entity2mention, entity2mention_standard):
            assert x.lower() == y.lower()
            wiki = ""
            if x in linked_entities:
                wiki = linked_entities[x] 
            entity2mention[x] = (entity2mention[x], wiki)
            entity2mention_standard[y] = (entity2mention_standard[y], wiki)

        # Bring lowercase back to standard:
        new_entity_list_ref = dict()
        for x in new_entity_list:
            ann_idx = annotator_entities.index(x)
            ann_orig = annotator_entities_standard[ann_idx]
            mapped_entity_idx = list(entity2mention.keys()).index(new_entity_list[x][0])
            mapped_entity = list(entity2mention_standard.keys())[mapped_entity_idx]

            new_entity_list_ref[ann_orig] = (mapped_entity, new_entity_list[x][1])

        # a dictionary: annotator mention => auto-entity
        entity_list_collection.append(new_entity_list_ref)  
        # all possible auto-entities w/ corresponding mentions and spans
        possible_entities_collection.append(entity2mention_standard)



    return  entity_list_collection, possible_entities_collection


