from nltk.translate.bleu_score import sentence_bleu

import subprocess
import re
import argparse
import os
import math
import bert_score 


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.basicConfig(level='ERROR')
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
import numpy as np 
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="metrics")

parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--clsf_name", type=str, default="textattack/bert-base-uncased-yelp-polarity")
parser.add_argument("--ext_clsf_name", type=str, default="textattack/bert-base-uncased-yelp-polarity")
parser.add_argument("--attr_file", type=str, default="/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/data/yelp/test_li.attr")
parser.add_argument("--text_file", type=str, default="/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/data/yelp/test_li.txt")
parser.add_argument("--ref_file", type=str, default="/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/data/yelp/test_li_reference.txt")
parser.add_argument("--ext_clsf", action="store_true")
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--lstm_clsf", action="store_true")
parser.add_argument("--form_em_lstm", action="store_true")




args = parser.parse_args()


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

CLS = "[CLS]"
SEP = "[SEP]"
EOT_TOKEN = '<|endoftext|>'



def make_gpt_command_dom(checkpt_dir):

    com = r"""/home/NAME/PROJECT/Deep-Learning/GPT2-HarryPotter-Training/examples/run_lm_finetuning.py
        --output_dir=/home/NAME/PROJECT/Deep-Learning/GPT2-HarryPotter-Training/examples/output-dom{}
        --model_type=gpt2
        --model_name_or_path=gpt2-medium
        --train_data_file=/home/NAME/PROJECT/style-pooling/{}/dev.trans_{}a{}
        --do_eval
        --eval_data_file=/home/NAME/PROJECT/style-pooling/{}/dev.trans_{}a{}
        --block_size=200
        --per_gpu_train_batch_size=1""".format(dom,dire,step,alph,dire,step,alph)
    return com


def get_gpt_ppl(checkpoint_dir, file_name='opt_samples.txt',):
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  
    
    ppl_list = []
    sent_list = ""
    with open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
        for line in input_file:
            
            text = line[:-1]
            sent_list_temp = sent_list +" "+text
            if len(sent_list_temp.split()) > 950 and len(sent_list_temp.split()) < 1000: 

                sent_list = sent_list_temp
                encoded_input = gpt_tokenizer(text, padding=True, return_tensors='pt').to(device)
                tokens = encoded_input['input_ids']
                target_ids = tokens.clone()
                loss = gpt_model(tokens, labels = target_ids)
                ppl_list.append(math.exp(loss[0]))
                sent_list = ""
            elif len(sent_list_temp.split())<950:
                sent_list = sent_list_temp
                continue
            else: # > 450
                encoded_input = gpt_tokenizer(text, padding=True, return_tensors='pt').to(device)
                tokens = encoded_input['input_ids']
                target_ids = tokens.clone()
                loss = gpt_model(tokens, labels = target_ids)
                ppl_list.append(math.exp(loss[0]))
                sent_list = text
                
                
    
    return np.mean(np.array(ppl_list))



def perplexity_fudge(checkpoint_dir,file_name='opt_samples.txt'):
    # calculate perplexity 
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        with open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
                ppl = []
                sos_token = gpt_tokenizer.decode([0])
                for line in input_file:
                    sentence=line[:-1]
                    sentence=sentence.lower()
                    sentence = tokenizer.tokenize(sentence)
                    sentence = tokenizer.convert_tokens_to_ids(sentence)
                    sentence = tokenizer.decode(sentence)
                    sentence=sentence.replace(' \' ','\'')
                    sentence=sentence.replace(' - ','-')
                    sentence=sentence.replace(' .','.')
                    sentence=sentence.replace(' ,',',')
                    
                    full_tensor_input = gpt_tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
                    full_loss = gpt_model(full_tensor_input, labels=full_tensor_input)[0].mean()
                    ppl.append(torch.exp(full_loss).flatten().cpu().item())
                    #print(sentence,torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl), np.std(ppl), np.median(ppl)




  
def energy_score_mlm(batch, model, mask_id,beta=1 ):
  #encoded_input = tokenizer(text, return_tensors='pt').to(device)
  #tokens = encoded_input['input_ids'][0]
  seq_len = len(batch[0])-2
  posns = [i+1 for i in range(seq_len)]
  #random.shuffle(posns)
  norm_score = [0.0] * batch.shape[0]
  raw_score = [0.0] * batch.shape[0]
  for posn in posns:
    old_wrd = batch[:,posn].clone()
    batch[:,posn] = mask_id
    output = model(batch)['logits'][:,posn,:]
    #print(output.shape)
    norm_output = output.log_softmax(dim=-1)
    for i in range(batch.shape[0]): #TODO check this
        raw_score[i] += output[i,old_wrd[i]].item()
        norm_score[i] += norm_output[i,old_wrd[i]].item()
    #raw_score += output[old_wrd].item()
    #norm_score += norm_output[old_wrd].item()
    batch[:,posn] = old_wrd
  return [-1.0*raw_s*beta for raw_s in raw_score], [-1.0*norm_s*beta for norm_s in norm_score]

def get_mlm_score(checkpoint_dir,file_name='opt_samples.txt'):
    # calculate perplexity 
    
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MASK = "[MASK]"
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    with torch.no_grad():
        with open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
                ppl = []
                for line in input_file:
                    sentence=line[:-1]
                    sentence=sentence.lower()
           
                    #sentence=sentence.replace(' \' ','\'')
                    #sentence=sentence.replace(' - ','-')
                    #sentence=sentence.replace(' .','.')
                    #sentence=sentence.replace(' ,',',')
                    
                    seed_text = tokenizer.tokenize(sentence)
                    batch = torch.tensor(get_init_text(seed_text, tokenizer_disc=tokenizer, max_len=15, batch_size=1)).to(device)
                    energy , _= energy_score_mlm(batch,model,mask_id)
                    ppl.append(energy[0])
                    #print(sentence,torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl), np.std(ppl), np.median(ppl)





def get_external_cls(checkpoint_dir, disc_name, file_name='opt_samples.txt', out_file_name='opt_cls_ext.txt'):
    tokenizer_disc = BertTokenizer.from_pretrained(disc_name)  # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
    model_disc = AutoModelForSequenceClassification.from_pretrained(disc_name).to(device) #("textattack/bert-base-uncased-imdb").to(device)
   
   


    with open(f"{checkpoint_dir}/{out_file_name}", "w+") as f_pred_attr,open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
        for i,line in enumerate((input_file)):
            
            seed_text = line[:-1].lower()
            #seed_text = seed_text.split('\t')[1]
            #print(seed_text)

            seed_text = tokenizer_disc.tokenize(seed_text)
            batch = torch.tensor(get_init_text(seed_text, tokenizer_disc=tokenizer_disc, max_len=15, batch_size=1)).to(device)
            #print(batch.shape)
            pred = disc_cls(batch,model_disc,tokenizer_disc)
            #print (pred[0])
            f_pred_attr.write(str(pred[0])+'\n')
            f_pred_attr.flush()
   
   
   
   

def get_hamming_dist_len_diff(checkpoint_dir, disc_name, src_file, file_name='opt_samples.txt', out_file_name='opt_cls_ext.txt'):
    tokenizer_disc = BertTokenizer.from_pretrained(disc_name)  # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
   
    cnt= 0
    dist =0
    
    len_diff_sum =0
   
    list_hammings =[]
    list_lens = []


    with open(f"{checkpoint_dir}/{file_name}", "r") as input_file, open(f"{src_file}", "r") as src_file:
        for i,(line, line_src) in enumerate(zip(input_file, src_file)):
            
            text = line[:-1].lower()
            line_src = line_src[:-1].lower()
            #seed_text = seed_text.split('\t')[1]
            #print(seed_text)

            text = tokenizer_disc.tokenize(text)
            line_src = tokenizer_disc.tokenize(line_src)
            
            if len(text) == len(line_src):
                dist += sum([a !=b for (a,b) in zip(text,line_src)  ])
                list_hammings.append(sum([a !=b for (a,b) in zip(text,line_src)  ]))
                list_lens.append(0)
            
            else:
                dist_2 =max(len(text), len(line_src))
                dist += max(len(text), len(line_src))
                
                
                if len(line_src) > len(text):
                    for element in text:
                            if element in line_src:
                                dist -= 1
                                dist_2 -=1
                else:
                    for element in line_src:
                        if element in text:
                            dist -= 1
                            dist_2 -=1
                
                
                len_diff_sum += abs(len(text)-len(line_src))
                
                list_hammings.append(dist_2)
                list_lens.append(abs(len(text)-len(line_src)))
                
                
            
            
            cnt+= 1
    
    #print(dist,len_diff_sum)
    return dist/cnt, len_diff_sum/cnt, list_hammings, list_lens
            
            
               
def get_lstm_cls(checkpoint_dir, disc_name, file_name='opt_samples.txt'):
    if not args.form_em_lstm:
        lstm_str  = subprocess.getoutput("bash ../batched_MH/scripts/jx_cls/run_clsf.sh {0} {1}".format(checkpoint_dir, disc_name))  
    else:
        lstm_str  = subprocess.getoutput("bash ../batched_MH/scripts/jx_cls/run_clsf_form_em.sh {0} {1}".format(checkpoint_dir, disc_name))  
        
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

def get_bert_score(src_file,transfile):
    bert_scorer = bert_score.BERTScorer(use_fast_tokenizer=True, lang='en')
    avg =0
    cnt = 0
    with open(src_file, "r") as src , open(f"{transfile}/opt_samples.txt", "r") as trans :
        for src_sample, trans_sample in zip (src,trans):
            #print(src_sample[:-1],trans_sample[:-1])
            P, R, F1 = bert_scorer.score([trans_sample[:-1]], [src_sample[:-1]], verbose=False, batch_size=1)
            #score = sentence_bleu(src_sample[:-1], trans_sample[:-1])
            #print(score)
            avg += F1
            cnt += 1

    #print("bleu is:", avg/cnt)
    #print(cnt, avg)
    
    return avg.item()/cnt


checkpoint_dir = args.checkpoint_dir
#"/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/output_samples/yelp_imdb/\
#disc_yelp_data_yelp_li_test_max_iter_8_temp_1.0_shuffle_True_block_False_alpha_0.0_beta_1.0_delta_0.0_date_22_10_2021_17_11_21"

#checkpoint_dir ="/home/user/dir_projects/dir.bert_sample/sent_transfer_discrim/deep-latent-sequence-model/model_outputs/yelp/deep_latent_seq"

#

attr_file = args.attr_file
txt_file= args.text_file
ref_file= args.ref_file


#cnt, acc,acc_to_0,acc_to_1 = 0,0,0,0 #get_cls_scores(attr_file,checkpoint_dir,file_name='opt_cls.txt', args.reverse) TODO
acc_ext,acc_to_0_ext, acc_to_1_ext = 1, 1, 1

#internal clsf
#if not os.path.exists(f'{checkpoint_dir}/opt_cls_met.txt'):

get_external_cls(checkpoint_dir, args.clsf_name, file_name='opt_samples.txt',out_file_name='opt_cls_met.txt')

#elif  sum(1 for line in open(f'{checkpoint_dir}/opt_cls_met.txt')) != sum(1 for line in open(f'{checkpoint_dir}/opt_samples.txt')):
#    get_external_cls(checkpoint_dir, args.clsf_name, file_name='opt_samples.txt',out_file_name='opt_cls_met.txt')

cnt, acc,acc_to_0,acc_to_1  =  get_cls_scores(attr_file,checkpoint_dir,file_name='opt_cls_met.txt') #1,1,1, 1 #



#external clsf
if args.ext_clsf or args.lstm_clsf:
    if not os.path.exists(f'{checkpoint_dir}/opt_cls_ext.txt'):
        if args.lstm_clsf:
            get_lstm_cls(checkpoint_dir, args.ext_clsf_name, file_name='opt_samples.txt')
        else: 
            get_external_cls(checkpoint_dir, args.ext_clsf_name, file_name='opt_samples.txt')
    elif  True: #sum(1 for line in open(f'{checkpoint_dir}/opt_cls_ext.txt')) != cnt: TODO
        if args.lstm_clsf:
            get_lstm_cls(checkpoint_dir, args.ext_clsf_name, file_name='opt_samples.txt')
        else: 
            get_external_cls(checkpoint_dir, args.ext_clsf_name, file_name='opt_samples.txt')
    
    cnt,acc_ext,acc_to_0_ext, acc_to_1_ext =  get_cls_scores(attr_file,checkpoint_dir,file_name='opt_cls_ext.txt', reverse = args.reverse)




####bleu
bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(ref_file, checkpoint_dir+"/opt_samples.txt"))



self_bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(txt_file, checkpoint_dir+"/opt_samples.txt"))


bleu_str =  bleu_str.split('\n')[-1].strip()
reg = re.compile("BLEU = ([^,]*).*")
try:
    perl_bleu = float(reg.match(bleu_str).group(1))
except:
    perl_bleu = 0.


self_bleu_str =  self_bleu_str.split('\n')[-1].strip()
reg = re.compile("BLEU = ([^,]*).*")
try:
    self_perl_bleu = float(reg.match(self_bleu_str).group(1))
except:
    self_perl_bleu = 0.



####hamming 
dist_ham, len_diff , list_hams, list_dist = get_hamming_dist_len_diff(checkpoint_dir, args.clsf_name, src_file=args.text_file)

self_bleu = 0 #get_bleu_score(txt_file,checkpoint_dir) TODO
bleu = 0 #get_bleu_score(ref_file,checkpoint_dir) TODO multiple

####gpt
mlm_mean_score = get_mlm_score(checkpoint_dir,file_name='opt_samples.txt')
gpt_mean_score = perplexity_fudge(checkpoint_dir,file_name='opt_samples.txt')

###Bertscore
self_bertsc = get_bert_score(txt_file,checkpoint_dir) 
bertsc = get_bert_score(ref_file,checkpoint_dir) 




with open(f"{checkpoint_dir}/metrics.txt", "w+") as file:
    file.write(','.join([str(acc),str(bleu),str(self_bleu),str(perl_bleu/100),str(self_perl_bleu/100), str(acc_ext),str(acc_to_0_ext),str(acc_to_1_ext), str(cnt),str(acc_to_1),str(acc_to_0),str( dist_ham),str(len_diff),str(gpt_mean_score) ,str(mlm_mean_score),str(self_bertsc), str(bertsc) ,str(np.std( np.array(list_hams))),str(np.std(np.array(list_dist)))]))
    file.flush()
    file.close()
    print(','.join([str(acc),str(bleu),str(self_bleu),str(perl_bleu/100),str(self_perl_bleu/100), str(acc_ext),str(acc_to_0_ext),str(acc_to_1_ext),str(cnt),str(acc_to_1),str(acc_to_0),str( dist_ham),str(len_diff), str(gpt_mean_score),str(mlm_mean_score) ,str(self_bertsc), str(bertsc) , str(np.std(np.array( list_hams))),str(np.std(np.array(list_dist)))]))
    #print(','.join([str(acc),str(bleu),str(self_bleu),str(perl_bleu/100),str(self_perl_bleu/100),str(cnt),str(acc_to_1),str(acc_to_0)]))
        



