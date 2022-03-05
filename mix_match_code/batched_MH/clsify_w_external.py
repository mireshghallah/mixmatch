

import os
import random

import numpy as np
from numpy.core.fromnumeric import shape
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from torch.distributions.categorical import Categorical
from datetime import datetime
import random





disc_dir= "textattack/bert-base-uncased-yelp-polarity"
#text_dir="/home/user/dir_projects/dir.bert_sample/sent_transfer_discrim/deep-latent-sequence-model/model_outputs/yelp/deep_latent_seq"
text_dir='/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/output_samples/yelp_imdb/\
disc_yelp_data_yelp_li_test_max_iter_8_temp_1.0_shuffle_True_block_False_alpha_50.0_beta_1.0_delta_40.0_date_24_10_2021_05_10_48'

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'


tokenizer_disc = AutoTokenizer.from_pretrained(disc_dir)  # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
model_disc = AutoModelForSequenceClassification.from_pretrained(disc_dir).to(device) #("textattack/bert-base-uncased-imdb").to(device)


CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer_disc.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer_disc.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer_disc.convert_tokens_to_ids([CLS])[0]


def tokenize_batch(batch):
    return [tokenizer_disc.convert_tokens_to_ids(sent) for sent in batch]



def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [ [CLS]+ seed_text  + [SEP]  for _ in range(batch_size)] #TODO

    return tokenize_batch(batch)

def disc_cls(batch,model_disc,tokenizer_disc):
  #encoded_input = tokenizer(text, return_tensors='pt').to(device)
  #tokens = encoded_input['input_ids'][0]

  output = model_disc(batch)['logits']
  pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()),axis=-1)

  #print(output.shape)

  return  pred





with open(f"{text_dir}/opt_cls_ext.txt", "w+") as f_pred_attr,open(f"{text_dir}/opt_samples.txt", "r") as input_file:
    for i,line in enumerate((input_file)):
        
        seed_text = line[:-1]
        #seed_text = seed_text.split('\t')[1]
        print(seed_text)

        seed_text = tokenizer_disc.tokenize(seed_text)
        batch = torch.tensor(get_init_text(seed_text, max_len=15, batch_size=1)).to(device)
        #print(batch.shape)
        pred = disc_cls(batch,model_disc,tokenizer_disc)
        #print (pred[0])
        f_pred_attr.write(str(pred[0])+'\n')
        f_pred_attr.flush()
        



        