## Edited and reused from  https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb

import os
import random

import numpy as np
from numpy.core.fromnumeric import shape
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertTokenizer, BertModel
from torch.distributions.categorical import Categorical
from datetime import datetime
import random
import bert_score 

import argparse
import warnings

warnings.filterwarnings("ignore")

################

parser = argparse.ArgumentParser(description="style transfer")

parser.add_argument("--max_iter", type=int, help="number of changes to make in the gibbs chain", default=100)
parser.add_argument("--n_samples", type=int, help="number of changes to make in the gibbs chain", default=20)
parser.add_argument("--batch_size", type=int, help="number of changes to make in the gibbs chain", default=20)

parser.add_argument("--temperature", type=float, help="number of changes to make in the gibbs chain", default=1.0)
parser.add_argument("--degenerate", action='store_true')
parser.add_argument("--block", action='store_true')
parser.add_argument("--shuffle_positions", action='store_true')


###degenerate gibbs sampler
parser.add_argument("--top_k", type=int, help="top_k sampler-so far only degenerate support", default=40)
parser.add_argument("--burnin", type=int, help="burn in for degenerate support", default=250)


parser.add_argument("--data_path", type=str, help="dir", default='./data/yelp')
parser.add_argument("--attr_path", type=str, help="dir", default='./data/yelp')


parser.add_argument("--data_name", type=str, help="dir", default='yelp')


parser.add_argument("--out_path", type=str, help="dir", default='./batched')
parser.add_argument("--model_path", type=str, help="dir", default='bert-base-uncased')
parser.add_argument("--tok_path", type=str, help="dir", default='bert-base-uncased')

#disc 

parser.add_argument("--disc_name", type=str, help="disc dir", default='imdb')
parser.add_argument("--disc_dir", type=str, help="disc dir", default='textattack/bert-base-uncased-imdb')

#hyper params
parser.add_argument("--alpha", type=float, help="knob", default=1) # disc
parser.add_argument("--beta", type=float, help="knob", default=1)
parser.add_argument("--delta", type=float, help="knob", default=1) # hamming
parser.add_argument("--gamma", type=float, help="knob", default=0) #bluert score
parser.add_argument("--theta", type=float, help="knob", default=0) #bertscore




args = parser.parse_args()


##################

cuda = torch.cuda.is_available()
print(cuda)
device = 'cuda' if cuda else 'cpu'

# Load pre-trained model (weights)
model_version = args.model_path #os.environ["MODEL_PATH"]
model = AutoModelForMaskedLM.from_pretrained(model_version)
model.eval()

if cuda:
    model = model.cuda()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(args.tok_path)



CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
PAD = "[PAD]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
#mr_id = 2720 #tokenizer.convert_tokens_to_ids("mr")[0]
#ms_id = 5796 #tokenizer.convert_tokens_to_ids("ms")[0]



if args.gamma :
    model_bleurt = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    model_bleurt.eval()
    
    if cuda:
        model_bleurt = model_bleurt.cuda()

if args.theta :
    bert_scorer = bert_score.BERTScorer(use_fast_tokenizer=True, lang='en')
    
    


def get_opt_sent(sents,metadata):
    min_score = 10000
    ind = 0
    meta_array = np.array(metadata)
    
    ind = np.argmin(meta_array[:,1,...])
    val = np.min(meta_array[:,1,...])
    sent_best = sents[ind].split()
    return " ".join(sent_best[1:-1]), meta_data[ind][-4],ind


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(list(sent.to('cpu').numpy())) for sent in batch]


def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent




def get_bert_score(batch, seed_text):
    sents = untokenize_batch(batch)
    
    sents = [(" ".join(item[1:-1])).strip() for item in sents] 
    
    #print(sents, seed_text)
    P, R, F1 = bert_scorer.score(sents, seed_text, verbose=False, batch_size=args.batch_size)
    #print(F1)         
    
    #print(np.array(cnt))
    return np.array(R) #high recall
                


def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, pad_len = 0, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [ [CLS]+ seed_text  + [SEP] +[PAD]*pad_len  for _ in range(batch_size)] #TODO

    return tokenize_batch(batch)


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def to_file(sents, file):
    with open(file, "a") as f:
        f.write("\n".join(sents) + "\n")


# Generation modes as functions
import math
import time



#   return  score
  
def energy_score_mlm(batch,beta = 1):
  #encoded_input = tokenizer(text, return_tensors='pt').to(device)
  #tokens = encoded_input['input_ids'][0]
  seq_len = len(batch[0])-2
  posns = [i+1 for i in range(seq_len)]
  #random.shuffle(posns)
  norm_score = [0.0] * batch.shape[0]
  raw_score = [0.0] * batch.shape[0]
  for posn in posns:
    old_wrd = batch[:,posn].clone()
    #print(tokenizer.decode(tokens[1:-1]))
    batch[:,posn] = mask_id
    #output = model(**encoded_input)[0][0,posn,:].log_softmax(dim=-1)
    #output = model(**encoded_input)[0][0,posn,:]
    #print(batch.shape, posn)
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

def energy_score_disc(batch,model_disc,tokenizer_disc,sentiment=0, alpha=0):
  #encoded_input = tokenizer(text, return_tensors='pt').to(device)
  #tokens = encoded_input['input_ids'][0]
  seq_len = len(batch[0])-2
  posns = [i+1 for i in range(seq_len)]
  #random.shuffle(posns)
  norm_score = np.array([0.0] * batch.shape[0])
  raw_score = np.array([0.0] * batch.shape[0])
  
  output = model_disc(batch)['logits']
  pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()),axis=-1)

  #print(output.shape)
  classes = output.shape[-1]
  for i in range (classes):
    if i == sentiment:
      raw_score += np.array(output[:,i].cpu().detach())
      norm_output = output.log_softmax(dim=-1)
      norm_score += np.array(norm_output[:,i].cpu().detach())



  return [-1.0*raw_s*alpha for raw_s in raw_score], [-1.0*norm_s*alpha for norm_s in norm_score], pred


def parallel_sequential_generation(
    seed_text,
    model_disc,
    tokenizer_disc,
    sentiment = 0,
    batch_size=10,
    max_len=15,
    top_k=0,
    temperature=1,
    max_iter=300,
    burnin=200,
    cuda=False,
    print_every=10,
    verbose=True,
    args=args
):
    """Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    #     seed_len = len(seed_text)
    #if (sentiment == 0) and (len(seed_text) > 12):
    #    max_len =  random.randint(10,len(seed_text))  #len(seed_text) -  5
    #else:
    #    max_len = len(seed_text)
        
    max_len = len(seed_text)
    
    batch = torch.tensor(get_init_text(seed_text[:max_len], max_len,pad_len=0, batch_size = batch_size)).to(device)

    batch_original = torch.tensor(get_init_text(seed_text[:max_len], max_len, pad_len = 0, batch_size=batch_size)).to(device)  #batch.detach().clone()



    seed_text_j = (" ".join(seed_text)).strip()
    seed_text_broad=[seed_text_j for i in range(batch.shape[0])]
        
    seq_len=max_len #batch.shape[-1]-2
    posns = [i+1 for i in range(seq_len)]
    #posns =  [i for i, y in enumerate(batch[0])  if (y == mask_id or y == mr_id or y==ms_id)] #TODO [i for i, y in enumerate(batch[0]) if y == mask_id]
    #print(batch)
    #print(mask_pos)

    full_meta_data =[[] for i in range(batch_size)]
    meta_data =[]
    for ii in range(max_iter):
        if (args.shuffle_positions):
            random.shuffle(posns)   
        if not (args.block):
            nmasks = 1
        else:
            nmasks= random.randint(max(1,math.ceil(seq_len/2)-3), min(seq_len-1, math.ceil(seq_len/2)+3))
        
        


        groups = [posns[i:i+nmasks] for i in range(0, len(posns), nmasks)]
        if (args.shuffle_positions):
            random.shuffle(groups) 
        #kk = mask_pos[np.random.randint(0, len(mask_pos))]
        for positions in groups:
            
            if args.degenerate: 
                # for jj in range(batch_size):
                #     batch[jj][kk] = mask_id
                # inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
                # out = model(inp)
                # topk = top_k if (ii >= burnin) else 0
                # idxs = generate_step(
                #     out,
                #     gen_idx=kk,
                #     top_k=topk if (ii >= burnin) else 0,
                #     temperature=temperature,
                #     sample=(ii < burnin),
                # )
                # for jj in range(batch_size):
                #     batch[jj][kk] = idxs[jj]
                # r_score, norm_score = np.array(energy_score(batch))
                # for i in range(batch_size):
                #     meta_data[i].append( (ii,kk,r_score[i],norm_score[i]) )
                               #old_e = np.array(enorm_score(batch))
                               #print(kk)
                #old_e = np.array(enorm_score(batch))
                old_r, old_norm = np.array(energy_score_mlm(batch))
                #print(nmasks, positions, groups)
                old_wrd = batch[:,positions].detach().clone()
                
                batch[:,positions] = mask_id
                #print(old_wrd.shape)
                ##here
                #output = model(batch)[:,kk,:].softmax(dim=-1)
                output = (model(batch)[:,positions,:]/temperature)
                output[:,:,mask_id] = -10000000000.0
                output = output.softmax(dim=-1)

                #print(output.shape)
                qxbx = np.array([1.0]*batch_size)
                qxxb = np.array([1.0]*batch_size)
                #for i,posn in enumerate(positions):

                d = Categorical(output)
                new_wrd = d.sample()
                #print(new_wrd.shape)
                #n_flag = ~(old_wrd[i] == new_wrd) #TODO
                n_flag = np.array([0]*batch_size)
                msk_change = [False] * batch_size
                
                for ii in range(len(positions)):
                    for jj in range(batch_size):
                        #print("shape",output[:,ii,old_wrd[jj,ii]].cpu().shape)
                        qxxb[jj] *= output[jj,ii,old_wrd[jj,ii]].cpu()
                #    qxxb.append(output[jj,old_wrd[i][jj]].item()) 
                        qxbx[jj] *= output[jj,ii,new_wrd[jj,ii]].cpu()
                        if not(old_wrd[jj,ii].item() == new_wrd[jj,ii].item()):    
                            n_flag[jj] = 1
                        if (old_wrd[jj,ii].item() == mask_id):
                            msk_change[jj] = True 
                

                batch[:,positions] = new_wrd
                new_r,new_norm = np.array(energy_score_mlm(batch))
                
        
                #mask_id == np.array(old_wrd.cpu())
                
                #print(msk_change)
                axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide( np.multiply(np.exp(old_r - new_r),np.array(qxxb)),np.array(qxbx))) )
                
                #print(axbx.shape)
                
                acc = torch.ones(axbx.shape)#torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))
    

                batch[:,positions] = torch.where(acc.unsqueeze(1).repeat(1,len(positions)).to(device)>0.0,batch[:,positions],old_wrd) 
                
                r_score  = np.squeeze(np.where(acc>0.0,new_r,old_r) )
                norm_score =np.squeeze( np.where(acc>0.0,new_norm,old_norm) )


                acc = np.array(acc.cpu()) * np.array(n_flag)

                #for i in range(batch_size):
                #    meta_data[i].append( (ii,positions,r_score[i],norm_score[i],qxxb[i],qxbx[i],axbx[i],acc[i].item()) )
                    
   
                #print(meta_data)
                #exit(0)
   
                
            else:
                #print(kk)
                #old_e = np.array(enorm_score(batch))
                distance = 0
                if args.delta:
                    distance = np.sum(1-np.array((batch == batch_original).detach().cpu())*1,axis=-1)
                bleurt_score = 0 #np.zeros(distance.shape)
                if args.gamma:
                    bleurt_score =np.array( model_bleurt(batch_original,batch)[0].squeeze().detach().cpu() )
                
                bert_score = 0
                if args.theta:
                    bert_score=get_bert_score(batch,seed_text_broad)
                    #print(org_emb)
                    
                    #b_emb = 
                    #bert_sim_score =

                #print(distance,"dist before")
                disc_1,disc_2, disc_preds = energy_score_disc(batch,model_disc=model_disc,tokenizer_disc=tokenizer_disc,sentiment=sentiment,alpha=args.alpha)
                old_r, old_norm = np.array(energy_score_mlm(batch,args.beta))+np.array([disc_1,disc_2])
                old_r += args.delta*distance
                old_r -= args.gamma*bleurt_score
                old_r -= args.theta*bert_score
                #print(nmasks, positions, groups)
                old_wrd = batch[:,positions].detach().clone()
                
                batch[:,positions] = mask_id
                #print(old_wrd.shape)
                ##here
                #output = model(batch)[:,kk,:].softmax(dim=-1)
                output = (model(batch)['logits'][:,positions,:]/temperature)
                output[:,:,mask_id] = -10000000000.0
                output = output.softmax(dim=-1)

                #print(output.shape)
                qxbx = np.array([1.0]*batch_size)
                qxxb = np.array([1.0]*batch_size)
                #for i,posn in enumerate(positions):

                d = Categorical(output)
                new_wrd = d.sample()
                #print(new_wrd.shape)
                #n_flag = ~(old_wrd[i] == new_wrd) #TODO
                n_flag = np.array([0]*batch_size)
                msk_change = [False] * batch_size
                
                for ii in range(len(positions)):
                    for jj in range(batch_size):
                        #print("shape",output[:,ii,old_wrd[jj,ii]].cpu().shape)
                        qxxb[jj] *= output[jj,ii,old_wrd[jj,ii]].cpu()
                #    qxxb.append(output[jj,old_wrd[i][jj]].item()) 
                        qxbx[jj] *= output[jj,ii,new_wrd[jj,ii]].cpu()
                        if not(old_wrd[jj,ii].item() == new_wrd[jj,ii].item()):    
                            n_flag[jj] = 1
                        if (old_wrd[jj,ii].item() == mask_id):
                            msk_change[jj] = True 
                

                batch[:,positions] = new_wrd

                distance_new = 0
                if args.delta:
                    distance_new = np.sum(1-np.array((batch == batch_original).detach().cpu())*1,axis=-1)
                bleurt_new = 0
                if args.gamma:
                    bleurt_new = np.array( model_bleurt(batch_original,batch)[0].squeeze().detach().cpu() )
                
                bert_new =0
                if args.theta:
                    bert_new = get_bert_score(batch, seed_text_broad)
                #print("new dist\n",distance)
                disc_1,disc_2, disc_preds_new = energy_score_disc(batch,model_disc=model_disc,tokenizer_disc=tokenizer_disc,sentiment=sentiment,alpha=args.alpha)
                new_r,new_norm = np.array(energy_score_mlm(batch,beta=args.beta))+np.array([disc_1,disc_2])
                new_r += args.delta*distance_new
                new_r -= args.gamma*bleurt_new
                new_r -= args.theta*bert_new

        
                #mask_id == np.array(old_wrd.cpu())
                
                #print(msk_change)
                axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide( np.multiply(np.exp(old_r - new_r),np.array(qxxb)),np.array(qxbx))) )
                
                #print(axbx.shape)
                
                acc = torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))
    

                batch[:,positions] = torch.where(acc.unsqueeze(1).repeat(1,len(positions)).to(device)>0.0,batch[:,positions],old_wrd) 
                
                r_score  = np.squeeze(np.where(acc>0.0,new_r,old_r) )
                norm_score =np.squeeze( np.where(acc>0.0,new_norm,old_norm) )
                disc_preds  = np.squeeze(np.where(acc>0.0,disc_preds_new,disc_preds) )   
                distance  = np.squeeze(np.where(acc>0.0,distance_new,distance) ) 
                bleurt  = np.squeeze(np.where(acc>0.0,bleurt_new,bleurt_score) ) 
                bert_score  = np.squeeze(np.where(acc>0.0,bert_new,bert_score) ) 

                acc = np.array(acc.cpu()) * np.array(n_flag)

                for i in range(batch_size):
                    full_meta_data[i].append( (sentiment, r_score[i],norm_score[i],qxxb[i],qxbx[i],axbx[i],acc[i].item(),disc_preds[i], distance[i] ,bleurt[i], bert_score[i]))
                  
   
                #print(meta_data)
                #exit(0)


        if verbose and np.mod(ii + 1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[: kk + 1] + ["(*)"] + for_print[kk + 1 :]
            print("iter", ii + 1, " ".join(for_print))
    
    for i in range(batch_size):
        meta_data.append( (sentiment, r_score[i],norm_score[i],qxxb[i],qxbx[i],axbx[i],acc[i].item(),disc_preds[i] ,distance[i],bleurt[i],bert_score[i]))
        

    return untokenize_batch(batch),meta_data, full_meta_data


def generate(
    n_samples,
    model_disc,
    tokenizer_disc,
    sentiment,
    seed_text="[CLS]",
    batch_size=10,
    max_len=25,
    top_k=100,
    temperature=1.0,
    burnin=200,
    max_iter=500,
    cuda=False,
    print_every=1,
    args=args
):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        batch , metadata, full_metadata= parallel_sequential_generation(
            seed_text,
            model_disc=model_disc,
            tokenizer_disc=tokenizer_disc,
            batch_size=batch_size,
            sentiment=sentiment,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            verbose=False,
        )

        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()

        sentences += batch
    return sentences, metadata, full_metadata




# Choose the prefix context

seeds = [
    "[CLS] mr",  #TODO
    "[CLS] ms", #TODO
]

import secrets



#
degenerate = args.degenerate
#
top_k = args.top_k #40 #not used
#leed_out_len = 5  # max_len, not used
burnin = args.burnin #250 #not used
temperature = args.temperature
###########


dirname = args.out_path
n_samples = args.n_samples
batch_size = args.batch_size
max_iter = args.max_iter
max_len =1 #this is dummy!!
########
tokenizer_disc = AutoTokenizer.from_pretrained(args.disc_dir) #BertTokenizer # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
model_disc = AutoModelForSequenceClassification.from_pretrained(args.disc_dir).to(device) #("textattack/bert-base-uncased-imdb").to(device)



now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

if args.degenerate:
    folder_name = "degenerate_topk_{}_burnin_{}_disc_{}_data_{}_max_iter_{}_temp_{}_shuffle_{}_block_{}_alpha_{}_beta_{}_delta_{}_gamma_{}_theta_{}_date_{}".format(top_k,burnin, args.disc_name,args.data_name,max_iter,temperature,args.shuffle_positions,args.block,args.alpha,args.beta,args.delta, args.gamma, args.theta ,dt_string)    
else:
    folder_name = "disc_{}_data_{}_max_iter_{}_temp_{}_shuffle_{}_block_{}_alpha_{}_beta_{}_delta_{}_gamma_{}_theta_{}_date_{}".format(args.disc_name,args.data_name,max_iter,temperature,args.shuffle_positions,args.block, args.alpha,args.beta, args.delta,args.gamma, args.theta, dt_string)

directory = "{}/{}".format(dirname,folder_name)
if not os.path.exists(directory):
    os.mkdir(directory)

dirname=directory
data_dir = args.data_path
attr_dir = args.attr_path

with open(f"{dirname}/samples.txt", "w") as f, open(f"{dirname}/opt_samples.txt", "w") as optimal_f,open(f"{dirname}/opt_cls.txt", "w") as optimal_class,open(f"{dirname}/opt_meta.txt", "w") as opt_meta_file ,open(f"{dirname}/metadata.txt", "w") as f_meta , open(f"{data_dir}", "r") as data_file, open(f"{attr_dir}", "r") as attr_file:
    for i,(line, src) in enumerate(zip(data_file,attr_file)):
        
        seed_text = line[:-1]
        sentiment = int(1-int(src[:-1]))
        print(seed_text)
        seed_text = tokenizer.tokenize(seed_text)
        print(seed_text)
        torch.cuda.empty_cache()
        bert_sents, meta_data, full_meta_data = generate(
            n_samples,
            model_disc=model_disc,
            tokenizer_disc=tokenizer_disc,
            sentiment=sentiment,
            seed_text=seed_text,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            args=args
        )

        sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))

        f.write("\n".join(sents) + "\n")
        f.flush()

        
        #meta_data_str = [str(l) for l in meta_data]
        
        #f_meta.write("\n".join(meta_data_str)+"\n")
        #f_meta.flush()

        full_meta_data_str = [str(l) for l in full_meta_data]
        f_meta.write("\n".join(full_meta_data_str)+"\n")
        f_meta.flush


        opt_sent,opt_cls,ind = get_opt_sent(sents,meta_data)
        optimal_f.write(opt_sent + "\n")
        optimal_f.flush()


        opt_meta_str = str(full_meta_data[ind])
        opt_meta_file.write(opt_meta_str + "\n")
        opt_meta_file.flush()
        
        optimal_class.write(str(opt_cls) + "\n")
        optimal_class.flush()