

import os
import random

import numpy as np
from numpy.core.fromnumeric import shape
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from torch.distributions.categorical import Categorical
from datetime import datetime
import random





#disc_dir= "/home/user/dir_projects/dir.bert_sample/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-400"
text_dir="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/form_em/unmt"


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'






with open(f"{text_dir}/opt_samples.txt", "w") as f_text_out,open(f"{text_dir}/org_samples.txt", "r") as input_file:
    for i,line in enumerate((input_file)):
        
        seed_text = line[:-1]
        seed_text = seed_text.split('\t')[1]
        print(seed_text)
        f_text_out.write(seed_text + "\n")
        f_text_out.flush()




        