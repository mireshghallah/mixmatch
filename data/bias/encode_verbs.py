
from collections import defaultdict
import subprocess
import re
import argparse
import os

from pandas.core.accessor import delegate_names

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertTokenizer
import numpy as np 
import pandas as pd



parser = argparse.ArgumentParser(description="metrics")

parser.add_argument("--out_file", default="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/bias/verb_encoding.json" ,type=str)
parser.add_argument("--attr_file", type=str, default="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/bias/test_bi.attr")
parser.add_argument("--text_file", type=str, default="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/bias/test_mask.txt")

parser.add_argument("--agency_file", type=str, default="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/bias/agency_power.csv")





args = parser.parse_args()


def get_agency_dict(agency_file):
    
    dict_from_csv = pd.read_csv(agency_file, header=None, index_col=0, squeeze=True).to_dict()

    return dict_from_csv

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

CLS = "[CLS]"
SEP = "[SEP]"


agency_dict = get_agency_dict(args.agency_file)
#print(agency_dict[1].keys())
agency_dict = agency_dict[1]
agency_dict_main = {}
for key in agency_dict.keys():
    #print(key)
    agency_dict_main[key[:-1]]=agency_dict[key]

src = {'agency_neg':'0','agency_pos':'1', 'agency_equal':'2'}


from collections import defaultdict

agency_dict_main_encoded =defaultdict(list)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for i,key in enumerate(agency_dict_main.keys()):
    if i ==0 :
        continue
    tokens=tokenizer.tokenize(key)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    #print(agency_dict_main[key])
    #print(src[agency_dict_main[key]])  
    if agency_dict_main[key] == agency_dict_main[key] :  
        agency_dict_main_encoded[src[agency_dict_main[key]]].append(ids)
    

    
print(agency_dict_main_encoded.keys())

sentence = "she keeps herself through it"

sentence_toks = tokenizer.tokenize(sentence)
sent_ids  = tokenizer.convert_tokens_to_ids(sentence_toks)

print(sent_ids)
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("keeps")))

for element in agency_dict_main_encoded['1']:
    ids_cnt = len(element)
    for token in element:
        if int(token)  in sent_ids:
            ids_cnt -= 1
    if ids_cnt == 0:
        print(element)

  
exit(0)        

with open(args.input_file,'r') as inp_f, open(args.text_file, 'r') as text_f, open(args.attr_file,'r') as attr_file:
    for line, line_mask , attr in zip(inp_f,text_f,attr_file):
        agency_list =[]
        agency_list_mask =[]
        for key in agency_dict_main:
            if key in line and key not in line_mask :
                agency_list.append(agency_dict_main[key])
        if len(agency_list) != 0:
            cnt +=1
            if target_dict[attr[:-1]] in agency_list:
                cnt_corr +=1
        #else:
            #print(line, line_mask)

            
                
                    