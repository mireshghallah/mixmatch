from nltk.translate.bleu_score import sentence_bleu

import subprocess
import re
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertTokenizer
import numpy as np 
import pandas as pd



parser = argparse.ArgumentParser(description="metrics")

parser.add_argument("--input_file", default="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/bias/\
disc_bias_1050_data_bias_test_bi_boost_max_iter_15_temp_1.0_shuffle_True_block_False_alpha_100.0_beta_1.0_delta_50.0_gamma_0.0_theta_0.0_date_14_11_2021_14_35_42/opt_samples.txt" ,type=str)
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


def get_external_cls(checkpoint_dir, disc_name, file_name='opt_samples.txt', out_file_name='opt_cls_ext.txt'):
    tokenizer_disc = AutoTokenizer.from_pretrained(disc_name)  # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
    model_disc = AutoModelForSequenceClassification.from_pretrained(disc_name).to(device) #("textattack/bert-base-uncased-imdb").to(device)
   
   


    with open(f"{checkpoint_dir}/{out_file_name}", "w+") as f_pred_attr,open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
        for i,line in enumerate((input_file)):
            
            seed_text = line[:-1]
            #seed_text = seed_text.split('\t')[1]
            #print(seed_text)

            seed_text = tokenizer_disc.tokenize(seed_text)
            batch = torch.tensor(get_init_text(seed_text, tokenizer_disc=tokenizer_disc, max_len=15, batch_size=1)).to(device)
            #print(batch.shape)
            pred = disc_cls(batch,model_disc,tokenizer_disc)
            #print (pred[0])
            f_pred_attr.write(str(pred[0])+'\n')
            f_pred_attr.flush()
   
def get_lstm_cls(checkpoint_dir, disc_name, file_name='opt_samples.txt'):
    lstm_str  = subprocess.getoutput("bash ../batched_MH/scripts/jx_cls/run_clsf.sh {0} {1}".format(checkpoint_dir, disc_name))  
    print(lstm_str)
    #print(lstm_str)

def tokenize_batch(batch,tokenizer_disc):
    return [tokenizer_disc.convert_tokens_to_ids(sent) for sent in batch]



def get_init_text(seed_text, max_len, tokenizer_disc, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [ [CLS]+ seed_text  + [SEP]  for _ in range(batch_size)] #TODO

    return tokenize_batch(batch,tokenizer_disc)

def disc_cls(batch,model_disc,tokenizer_disc):
  #encoded_input = tokenizer(text, return_tensors='pt').to(device)
  #tokens = encoded_input['input_ids'][0]

  output = model_disc(batch)['logits']
  pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()),axis=-1)

  #print(output.shape)

  return  pred



        



def get_cls_scores(attr_file,transfered_attr,file_name='opt_cls.txt', reverse = False):
    all_sp_cor =[0,0]
    all_sp_cnt = [0,0]
    all_cor = 0
    all_cnt = 0

    #print(reverse)
    with open(attr_file, "r") as attr , open(f"{transfered_attr}/{file_name}", "r") as transfered_attr :
        for attr,trans in zip (attr,transfered_attr):
            trg = 1- int(attr[:-1])
            if reverse:
                
                tran= 1-int(trans[:-1])
                
            else:
                tran = int(trans[:-1])

            all_cnt +=1
            all_sp_cnt[trg] += 1


            if (tran == trg):
                all_cor +=1
                all_sp_cor[tran] += 1
            
    
    #print("overall acc: ", all_cor/all_cnt)
    #print("trans to class 0 acc:", all_sp_cor[0], all_sp_cnt[0])
    #print("trans to class 1 acc:", all_sp_cor[1], all_sp_cnt[1])
    _0_acc = all_sp_cor[0]/all_sp_cnt[0] if all_sp_cnt[0] else 0
    _1_acc = all_sp_cor[1]/all_sp_cnt[1] if all_sp_cnt[1] else 0
    return all_cnt,all_cor/all_cnt, _0_acc,_1_acc



def get_bleu_score(src_file,transfile):
    avg =0
    cnt = 0
    with open(src_file, "r") as src , open(f"{transfile}/opt_samples.txt", "r") as trans :
        for src_sample, trans_sample in zip (src,trans):
            #print(src_sample[:-1],trans_sample[:-1])
            score = sentence_bleu([src_sample[:-1].split()], trans_sample[:-1].split())
            #print(score)
            avg += score
            cnt += 1

    #print("bleu is:", avg/cnt)
    #print(cnt, avg)
    return avg/cnt



agency_dict = get_agency_dict(args.agency_file)
#print(agency_dict[1].keys())
agency_dict = agency_dict[1]
agency_dict_main = {}
for key in agency_dict.keys():
    #print(key)
    agency_dict_main[key[:-1]]=agency_dict[key]

target_dict = {'0':'agency_pos', '1':'agency_neg'}

cnt = 0
cnt_corr = 0

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

            
                
                    
        
 
    print(cnt_corr/cnt, cnt, cnt_corr)
        