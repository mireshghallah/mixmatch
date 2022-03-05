import numpy as np
import argparse
import time
import shutil
import gc
import random
import subprocess
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data_utils import DataUtil
from collections import defaultdict
from hparams import *
from utils import *
from model import *

class BiLSTMClassify(nn.Module):
    """docstring for BiLSTMClassify"""
    def __init__(self, hparams):
        super(BiLSTMClassify, self).__init__()
        self.hparams = hparams
        self.word_emb = nn.Embedding(self.hparams.src_vocab_size,
                                     self.hparams.d_word_vec,
                                     padding_idx=hparams.pad_id)

        self.lstm = nn.LSTM(self.hparams.d_word_vec,
                            self.hparams.d_model,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=2,
                            dropout=self.hparams.dropout)

        self.bridge = nn.Linear(hparams.d_model * 2, hparams.trg_vocab_size, bias=False)

        self.dropout = nn.Dropout(self.hparams.dropout)


    def forward(self, x_train, x_mask, x_len, step=None):
        batch_size, max_len = x_train.size()
        word_emb = self.word_emb(x_train)

        word_emb = self.dropout(word_emb)

        packed_word_emb = pack_padded_sequence(word_emb, x_len, batch_first=True)
        enc_output, (ht, ct) = self.lstm(packed_word_emb)
        enc_output, _ = pad_packed_sequence(enc_output, batch_first=True,
                                            padding_value=self.hparams.pad_id)

        # average pooling
        x_mask_neg = (1. - x_mask.float()).unsqueeze(-1)
        sent_embed = (enc_output * x_mask_neg).sum(1) / (x_mask_neg.sum(1))

        logits = self.bridge(sent_embed)

        return logits


class CNNClassify(nn.Module):

  def __init__(self, hparams):
    super(CNNClassify, self).__init__()
    self.hparams = hparams
    self.word_emb = nn.Embedding(self.hparams.src_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)

    self.conv_list = []
    self.mask_conv_list = []
    for c, k in zip(self.hparams.out_c_list, self.hparams.k_list):
      #self.conv_list.append(nn.Conv1d(self.hparams.d_word_vec, out_channels=c, kernel_size=k, padding = k // 2))
      self.conv_list.append(nn.Conv1d(self.hparams.d_word_vec, out_channels=c, kernel_size=k))
      nn.init.uniform_(self.conv_list[-1].weight, -args.init_range, args.init_range)
      self.mask_conv_list.append(nn.Conv1d(1, out_channels=c, kernel_size=k))
      nn.init.constant_(self.mask_conv_list[-1].weight, 1.0)

    self.conv_list = nn.ModuleList(self.conv_list)
    self.mask_conv_list = nn.ModuleList(self.mask_conv_list)
    for param in self.mask_conv_list.parameters():
      param.requires_grad = False

    self.project = nn.Linear(sum(self.hparams.out_c_list), self.hparams.trg_vocab_size, bias=False)
    nn.init.uniform_(self.project.weight, -args.init_range, args.init_range)
    if self.hparams.cuda:
      self.conv_list = self.conv_list.cuda()
      self.project = self.project.cuda()

  def forward(self, x_train, x_mask, x_len, step=None):
    batch_size, max_len = x_train.size()

    # [batch_size, max_len, d_word_vec]
    word_emb = self.word_emb(x_train)

    #x_mask = x_mask.unsqueeze(1).float()
    # [batch_size, d_word_vec, max_len]
    word_emb = word_emb.permute(0, 2, 1)
    conv_out = []
    for conv, m_conv in zip(self.conv_list, self.mask_conv_list):
      # [batch_size, c_out, max_len]
      c = conv(word_emb)
      #with torch.no_grad():
      #  m = m_conv(x_mask)
      #print(m_conv.weight)
      #print(m)
      #m = (m > 0)
      #print(m)
      #c.masked_fill_(m, -float("inf"))
      # [batch_size, c_out]
      c = c.max(dim=-1)
      conv_out.append(c[0])
    # [batch_size, trg_vocab_size]
    logits = self.project(torch.cat(conv_out, dim=-1))
    return logits

class simplenet(torch.nn.Module):
    def __init__(self, hparams):
        super(simplenet, self).__init__()
        self.hparams = hparams
        self.input_size = self.hparams.hidden_d
        self.num_classes  = self.hparams.trg_vocab_size
        self.hidden_size  = self.hparams.hidden_size
        self.hidden_size2  = self.hparams.hidden_size2
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size2, self.num_classes)
        #self.sigmoid = torch.nn.Sigmoid()
        self.learning_rate = self.hparams.lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu1(hidden)
        hidden = self.fc2(relu)
        relu = self.relu2(hidden)
        output = self.fc3(relu)
        #output = self.sigmoid(output)
        return output

class CNNClassifyWOEmb(nn.Module):

  def __init__(self, hparams):
    super(CNNClassifyWOEmb, self).__init__()
    self.hparams = hparams
    #print(self.hparams.trg_vocab_size, "vocab_size")
    self.conv_list = []
    self.mask_conv_list = []
    for c, k in zip(self.hparams.out_c_list, self.hparams.k_list):
      #self.conv_list.append(nn.Conv1d(self.hparams.d_word_vec, out_channels=c, kernel_size=k, padding = k // 2))
      self.conv_list.append(nn.Conv1d(self.hparams.hidden_d, out_channels=c, kernel_size=k))
      nn.init.uniform_(self.conv_list[-1].weight, -self.hparams.init_range, self.hparams.init_range)
      self.mask_conv_list.append(nn.Conv1d(1, out_channels=c, kernel_size=k))
      nn.init.constant_(self.mask_conv_list[-1].weight, 1.0)

    self.conv_list = nn.ModuleList(self.conv_list)
    self.mask_conv_list = nn.ModuleList(self.mask_conv_list)
    for param in self.mask_conv_list.parameters():
      param.requires_grad = False

    self.project = nn.Linear(sum(self.hparams.out_c_list), self.hparams.trg_vocab_size, bias=False)
    nn.init.uniform_(self.project.weight, -self.hparams.init_range, self.hparams.init_range)
    if self.hparams.cuda:
      self.conv_list = self.conv_list.cuda()
      self.project = self.project.cuda()

  def forward(self, x_train, step=None):
    #batch_size, max_len = x_train.size()
    # [batch_size, max_len, d_word_vec]
    #word_emb = self.word_emb(x_train)

    #x_mask = x_mask.unsqueeze(1).float()
    # [batch_size, d_word_vec, max_len]
    #word_emb = word_emb.permute(0, 2, 1)
    conv_out = []
    for conv, m_conv in zip(self.conv_list, self.mask_conv_list):
      # [batch_size, c_out, max_len]
      c = conv(x_train)
      #with torch.no_grad():
      #  m = m_conv(x_mask)
      #print(m_conv.weight)
      #print(m)
      #m = (m > 0)
      #print(m)
      #c.masked_fill_(m, -float("inf"))
      # [batch_size, c_out]
      c = c.max(dim=-1)
      conv_out.append(c[0])
    # [batch_size, trg_vocab_size]
    logits = self.project(torch.cat(conv_out, dim=-1))
    return logits


def test(model, data, hparams, test_src_file, test_trg_file, dir_out, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  file_out = open(f'{dir_out}/opt_cls_ext.txt','w+')
  data.reset_test(test_src_file, test_trg_file)
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt =  hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()
    for pred,label in zip(preds,labels):
      file_out.write(str(pred.detach().item())+"\n")
      file_out.flush()
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()
    if end_of_epoch:
      #print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      #print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      break
  return total_acc / file_count, total_loss


def test_nodoc_run_subset(model, data, hparams, test_src_file, test_trg_file, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0

  data.reset_test(test_src_file, test_trg_file)
  keep_list =[]
  with open('{}'.format(hparams.subset_list), 'r') as filehandle:
    keep_list = [int(current_place.rstrip()) for current_place in filehandle.readlines()]

  cnt = 0
  corr = 0
  all_sents = 0
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch

    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    labels = y.view(-1)
    for (logit, label, sample) in zip(logits, labels, x_len):
      if(cnt in keep_list):#cnt in keep_list
        if (np.argmax(logit.cpu().detach().numpy()) == label.cpu().detach().numpy()):
          corr +=1
        all_sents +=1
      cnt+=1

        

    if end_of_epoch:
      print("accuracy is ", float(corr)/float(all_sents))
      print (cnt)
      print(corr)
      print(all_sents)
      break
  return 


def test_run_avg_certainty(model, data, hparams, test_src_file, test_trg_file, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  cnt = 0

  diff_conf = 0
  log_probs = 0


  data.reset_test(test_src_file, test_trg_file)
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()
    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    #print("logits shape is", logits.shape)
    cnt += logits.shape[0]
    logits_np = torch.nn.functional.softmax(logits.view(-1, hparams.no_styles), dim=1).detach().cpu().numpy()

    # difference in confidence
    if hparams.no_styles == 2 :
      diff_conf +=np.sum(np.abs(logits_np[:,0] - logits_np[:,1]))
    if hparams.no_styles == 3 :
      diff_conf +=0#np.sum(np.abs(np.max(logits_np[:,0] , logits_np[:,1], logits_np[:,2]) - np.min(logits_np[:,0] , logits_np[:,1], logits_np[:,2]))) #diff_conf +=np.sum(np.abs(np.max(logits_np[:,0] , logits_np[:,1], logits_np[:,2]) - np.min(logits_np[:,0] , logits_np[:,1], logits_np[:,2])))
      
    #sum of log probs
    log_probs += np.sum(-np.multiply(np.log2(logits_np), logits_np))

    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()
    if end_of_epoch:
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      print("avg diff in certainty is", float(diff_conf)/float(cnt))
      print("avg log prob is ", float(log_probs)/float(cnt))
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      break
  return total_acc / file_count, total_loss


def test_run_count_certainty(model, data, hparams, test_src_file, test_trg_file, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  cnt = 0

  diff_conf = 0
  log_probs = 0


  data.reset_test(test_src_file, test_trg_file)
  cnt_classes = np.zeros(hparams.no_styles)
  corr_cnt_clasess = np.zeros(hparams.no_styles)
  
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()
    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    #print("logits shape is", logits.shape)
    #cnt += logits.shape[0]
    logits_np = torch.nn.functional.softmax(logits.view(-1, hparams.no_styles), dim=1).detach().cpu().numpy()


    # difference in confidence
    for (logit, label, sample) in zip(logits, y.view(-1), x_len):
      if hparams.no_styles == 3:
        thresh = 0.45
      else:
        thresh = 0.75
      if(max(logit)>thresh ):#0.7 for 2 doms    max(logit)>thresh  and sample<9
      #sentence_level_all += 1
        cnt_classes[np.argmax(logit.detach().cpu().numpy())] += 1
        if (np.argmax(logit.detach().cpu().numpy()) == label):
          corr_cnt_clasess[label] +=1 

      cnt+=1


    #sum of log probs
  

    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()
    if end_of_epoch:
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      print("count of confidence is: ",  np.sum(cnt_classes))
      print("confident accuracy is: ", np.sum(corr_cnt_clasess)/np.sum(cnt_classes) )
      print("ratio of confidence is: ",  np.sum(cnt_classes)/cnt)
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      break
  return total_acc / file_count, total_loss



def test_doc(model, data, hparams, test_src_file, test_trg_file, doc_dict, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0

  data.reset_test(test_src_file, test_trg_file)

  doc_logits = {}
  doc_votes = {}
  sentence_level_corr =0
  sentence_level_all = 0
  cnt = 0
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles
    logits = torch.nn.functional.softmax(logits.view(-1, targets_cnt), dim=1) #Fatemh:TODO hparams.trg_vocab_size
    labels = y.view(-1)
    
    for (logit, label, sample) in zip(logits, labels, x_len):
      
 
      if str(label.detach().item()) not in  doc_logits.keys():
        doc_logits[str(label.detach().item())] = logit.detach().cpu().numpy()
        doc_votes [str(label.detach().item())] = [0]*hparams.no_styles
        doc_votes [str(label.detach().item())][np.argmax(logit.detach().cpu().numpy())] += 1
        #print(np.argmax(doc_logits[str(label.detach().item())]))
      else:
        doc_logits[str(label.detach().item())]   += logit.detach().cpu().numpy()
        doc_votes [str(label.detach().item())][np.argmax(logit.detach().cpu().numpy())] += 1
   
      #print ((logit.detach().cpu().numpy()))

    #print(doc_logits)
    
    #if negate:
    #  labels = y_neg.view(-1)
    #else:
    #  labels = y.view(-1)
    #val_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    #_, preds = torch.max(logits, dim=1)
    #val_acc = torch.eq(preds, labels).int().sum()
    #n_batches += batch_size
    #valid_loss += val_loss.sum().item()
    #valid_acc += val_acc.item()
    if end_of_epoch:
      all_docs_cnt = 0
      corr_docs_cnt =0 
      for key in doc_logits.keys():
        all_docs_cnt += 1
        if doc_dict[key] == str(np.argmax(doc_logits[key])):
          corr_docs_cnt += 1
      print("doc level accuracy is ", float(corr_docs_cnt)/float(all_docs_cnt))

      all_docs_cnt = 0
      corr_docs_cnt =0 
      for key in doc_votes.keys():
        all_docs_cnt += 1
        if doc_dict[key] == str(np.argmax(doc_votes[key])):
          corr_docs_cnt += 1
        sentence_level_corr += doc_votes[key][int(doc_dict[key])]
        sentence_level_all += sum(doc_votes[key])
        #print(doc_votes[key])
      print("doc level accuracy using majority voting is ", float(corr_docs_cnt)/float(all_docs_cnt))
      print("sent level accuracy is ", float(sentence_level_corr)/float(sentence_level_all))
      #print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      #print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      #total_loss += valid_loss / n_batches
      #total_acc += valid_acc / n_batches
      #valid_words = 0
      #valid_loss = 0
      #valid_acc = 0
      #n_batches = 0
      #file_count += 1
      break
  return #total_acc / file_count, total_loss


def test_doc_save_subset(model, data, hparams, test_src_file, test_trg_file, doc_dict, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0

  data.reset_test(test_src_file, test_trg_file)

  doc_logits = {}
  doc_votes = {}
  sentence_level_corr =0
  sentence_level_all = 0
  cnt = 0
  keep_list = []
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles
    logits = torch.nn.functional.softmax(logits.view(-1, targets_cnt), dim=1) #Fatemh:TODO hparams.trg_vocab_size
    labels = y.view(-1)
    
    for (logit, label, sample) in zip(logits, labels, x_len):
      if hparams.no_styles == 3:
        thresh = 0.45
      else:
        thresh = 0.9
      if(max(logit)>thresh ):#0.7 for 2 doms    max(logit)>thresh  and sample<9
      #sentence_level_all += 1
        keep_list.append(cnt )
        if str(label.detach().item()) not in  doc_logits.keys():
          doc_logits[str(label.detach().item())] = logit.detach().cpu().numpy()
          doc_votes [str(label.detach().item())] = [0]*hparams.no_styles
          doc_votes [str(label.detach().item())][np.argmax(logit.detach().cpu().numpy())] += 1
          #print(np.argmax(doc_logits[str(label.detach().item())]))
        else:
          doc_logits[str(label.detach().item())]   += logit.detach().cpu().numpy()
          doc_votes [str(label.detach().item())][np.argmax(logit.detach().cpu().numpy())] += 1
      cnt+=1
      #print ((logit.detach().cpu().numpy()))


    if end_of_epoch:
      all_docs_cnt = 0
      corr_docs_cnt =0 
      for key in doc_logits.keys():
        all_docs_cnt += 1
        if doc_dict[key] == str(np.argmax(doc_logits[key])):
          corr_docs_cnt += 1
      print("doc level accuracy is ", float(corr_docs_cnt)/float(all_docs_cnt))

      all_docs_cnt = 0
      corr_docs_cnt =0 
      for key in doc_votes.keys():
        all_docs_cnt += 1
        if doc_dict[key] == str(np.argmax(doc_votes[key])):
          corr_docs_cnt += 1
        sentence_level_corr += doc_votes[key][int(doc_dict[key])]
        sentence_level_all += sum(doc_votes[key])
        #print(doc_votes[key])
      print("doc level accuracy using majority voting is ", float(corr_docs_cnt)/float(all_docs_cnt))
      print("sent level accuracy is ", float(sentence_level_corr)/float(sentence_level_all))
      print("sent level  answer rate is ", float(len(keep_list))/float(cnt))

      with open('{}'.format(hparams.subset_list), 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in keep_list)
      break
  return #total_acc / file_count, total_loss


def test_doc_run_subset(model, data, hparams, test_src_file, test_trg_file, doc_dict, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0

  data.reset_test(test_src_file, test_trg_file)

  doc_logits = {}
  doc_votes = {}
  sentence_level_corr =0
  sentence_level_all = 0
  cnt = 0
  all_sent_2 =0
  with open('{}'.format(hparams.subset_list), 'r') as filehandle:
      keep_list = [int(current_place.rstrip()) for current_place in filehandle.readlines()]

  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles
    logits = torch.nn.functional.softmax(logits.view(-1, targets_cnt), dim=1) #Fatemh:TODO hparams.trg_vocab_size
    labels = y.view(-1)
    
    for (logit, label, sample) in zip(logits, labels, x_len):
      if hparams.no_styles == 3:
        thresh = 0.45
      else:
        thresh = 0.75
      if(cnt in keep_list):#cnt in keep_list
        
        #keep_list.append(cnt)
        if ( True): #max(logit)>thresh
          all_sent_2 += 1
          if str(label.detach().item()) not in  doc_logits.keys():
            doc_logits[str(label.detach().item())] = logit.detach().cpu().numpy()
            doc_votes [str(label.detach().item())] = [0]*hparams.no_styles
            doc_votes [str(label.detach().item())][np.argmax(logit.detach().cpu().numpy())] += 1
            #print(np.argmax(doc_logits[str(label.detach().item())]))
          else:
            doc_logits[str(label.detach().item())]   += logit.detach().cpu().numpy()
            doc_votes [str(label.detach().item())][np.argmax(logit.detach().cpu().numpy())] += 1
      cnt+=1
      #print ((logit.detach().cpu().numpy()))


    if end_of_epoch:
      all_docs_cnt = 0
      corr_docs_cnt =0 
      for key in doc_logits.keys():
        all_docs_cnt += 1
        if doc_dict[key] == str(np.argmax(doc_logits[key])):
          corr_docs_cnt += 1
      print("doc level accuracy is ", float(corr_docs_cnt)/float(all_docs_cnt))

      all_docs_cnt = 0
      corr_docs_cnt =0 
      for key in doc_votes.keys():
        all_docs_cnt += 1
        if doc_dict[key] == str(np.argmax(doc_votes[key])):
          corr_docs_cnt += 1
        sentence_level_corr += doc_votes[key][int(doc_dict[key])]
        sentence_level_all += sum(doc_votes[key])
        #print(doc_votes[key])
      print("doc level accuracy using majority voting is ", float(corr_docs_cnt)/float(all_docs_cnt))
      print("sent level accuracy is ", float(sentence_level_corr)/float(sentence_level_all))
      print("sent level answer rate is ", float(all_sent_2)/float(cnt))


      break
  return #total_acc / file_count, total_loss


def test_confusion_matrix(model, data, hparams, test_src_file, test_trg_file, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  mat_full = [ [0]*hparams.no_styles for _ in range(hparams.no_styles) ]#[[0]*args.no_styles]*args.no_styles
  all_real_cnt = [0] * hparams.no_styles
  data.reset_test(test_src_file, test_trg_file)
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()
    

    for txt, pred_e , trg_e in zip(x, preds, labels):
      mat_full[trg_e.detach().item()][pred_e.detach().item()] +=1
      all_real_cnt[trg_e.detach().item()] += 1 
      

    if end_of_epoch:
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      break
  
  for i,row in enumerate(mat_full):
    print(row[0]/all_real_cnt[i], row[1]/all_real_cnt[i], row[2]/all_real_cnt[i])

  return total_acc / file_count, total_loss

def test_save_samples(model, data, hparams, test_src_file, test_trg_file, output_src_file, output_trg_file, output_decisions, corr_decisions, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0

  data.reset_test(test_src_file, test_trg_file)

  out_file_selection_src = open(output_src_file, 'w', encoding='utf-8')
  out_file_selection_trg = open(output_trg_file, 'w', encoding='utf-8')
  out_file_pred = open(output_decisions, 'w', encoding='utf-8')
  out_file_corr = open(corr_decisions, 'w', encoding='utf-8')
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()
    corr = torch.eq(preds, labels).int()
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()
    i=0
    #write specs:
    for  txt, corr_e, pred_e , trg_e in zip(x, corr, preds, labels):

      if hparams.reverse:
        if not corr_e:
          h_best_words = (map(lambda wi: data.src_i2w[wi],
                            filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id, hparams.pad_id], txt)))
          line =' '.join(h_best_words)
          line = line.strip()
          out_file_selection_src.write(line + '\n')
          out_file_selection_trg.write("dom"+str(trg_e.detach().item()) + '\n')
       
      else:
        if corr_e :
            h_best_words = (map(lambda wi: data.src_i2w[wi],
                              filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id, hparams.pad_id], txt)))
            line =' '.join(h_best_words)
            line = line.strip()
            out_file_selection_src.write(line + '\n')
            out_file_selection_trg.write("dom"+str(trg_e.detach().item()) + '\n')

      out_file_pred.write(str(pred_e.detach().item()) + '\n')
      out_file_corr.write(str(trg_e.detach().item()) + '\n')
      i += 1
      #print(corr_e.detach().item(), pred_e.detach().item(), trg_e.detach().item() )
    if(n_batches%10 == 0):
      print("Batch", n_batches) 

    if end_of_epoch:
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      break
  return total_acc / file_count, total_loss

def test_save_samples_numbers(model, data, hparams, test_src_file, test_trg_file, output_src_file, output_trg_file, output_decisions, corr_decisions,doc_dict, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  cnt = 0
  keep_list = []

  data.reset_test(test_src_file, test_trg_file)

  out_file_selection_src = open(output_src_file, 'w', encoding='utf-8')
  out_file_selection_trg = open(output_trg_file, 'w', encoding='utf-8')
  out_file_pred = open(output_decisions, 'w', encoding='utf-8')
  out_file_corr = open(corr_decisions, 'w', encoding='utf-8')
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = 0#torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = 0#torch.eq(preds, labels).int().sum()
    #corr = torch.eq(preds, labels).int()
    n_batches += batch_size
    valid_loss += 0#val_loss.sum().item()
    valid_acc += 0#val_acc.item()
    i=0
    #write specs:
    for  txt,  trg_e,pred_e, logit in zip(x, labels, preds,logits):
      corr_e = (pred_e.detach().item() == int(doc_dict[str(trg_e.detach().item())][-1]))
      if  (max(logit)>0.55 ): #corr_e  and 
          keep_list.append(i)
          h_best_words = (map(lambda wi: data.src_i2w[wi],
                            filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id, hparams.pad_id], txt)))
          line =' '.join(h_best_words)
          line = line.strip()
          out_file_selection_src.write(line + '\n')
          out_file_selection_trg.write(""+str(trg_e.detach().item()) + '\n')

      out_file_pred.write(str(pred_e.detach().item()) + '\n')
      out_file_corr.write(str(trg_e.detach().item()) + '\n')
      i += 1
      #print(corr_e.detach().item(), pred_e.detach().item(), trg_e.detach().item() )
    if(n_batches%10 == 0):
      print("Batch", n_batches) 

    if end_of_epoch:
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      with open('{}'.format(hparams.subset_list), 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in keep_list)
      break
  return total_acc / file_count, total_loss


def test_tpr_gap(model, data, hparams, test_src_file, test_trg_file, doc_dict_main, doc_dict_sens, negate=False):
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0

  doc_st_arts = {'Student':0, 'Arts':1} #{'dom1':0, 'dom0':1}#{'pos':0, 'neg':1} #
  cnt_all = 0
  corr_all = 0
  
  count_student_teen = 0
  count_student_teen_gstu = 0

  count_student_adult = 0
  count_student_adult_gstu = 0

  count_arts_teen = 0
  count_arts_teen_gsart = 0

  count_arts_adult = 0
  count_arts_adult_gsart = 0

  data.reset_test(test_src_file, test_trg_file)

  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_test(test_batch_size=hparams.valid_batch_size)
    # clear GPU memory
    #gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len)
    targets_cnt = hparams.no_styles if (hparams.no_styles) else hparams.trg_vocab_size
    logits = logits.view(-1, targets_cnt) #Fatemh:TODO hparams.trg_vocab_size
    if negate:
      labels = y_neg.view(-1)
    else:
      labels = y.view(-1)
    val_loss = 0#torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    _, preds = torch.max(logits, dim=1)
    val_acc = 0#torch.eq(preds, labels).int().sum()
    #corr = torch.eq(preds, labels).int()
    n_batches += batch_size
    valid_loss += 0#val_loss.sum().item()
    valid_acc += 0#val_acc.item()
   
    #write specs:
    for  txt,  trg_e,pred_e, logit in zip(x, labels, preds,logits):
      corr_e = (pred_e.detach().item() == int(doc_st_arts[doc_dict_main[str(trg_e.detach().item())]]))
      if (corr_e):
        corr_all += 1
      cnt_all += 1
      
      ####
      if (doc_dict_main[str(trg_e.detach().item())] == 'Student' and doc_dict_sens[str(trg_e.detach().item())] == 'dom0'):
        count_student_teen +=1
        if (corr_e):
          count_student_teen_gstu +=1
      elif (doc_dict_main[str(trg_e.detach().item())] == 'Student' and doc_dict_sens[str(trg_e.detach().item())] == 'dom1'):
        count_student_adult += 1
        if (corr_e):
          count_student_adult_gstu += 1
#####
      if (doc_dict_main[str(trg_e.detach().item())] == 'Arts' and doc_dict_sens[str(trg_e.detach().item())] == 'dom0'):
        count_arts_teen +=1
        if (corr_e):
          count_arts_teen_gsart +=1
      elif (doc_dict_main[str(trg_e.detach().item())] == 'Arts' and doc_dict_sens[str(trg_e.detach().item())] == 'dom1'):
        count_arts_adult += 1
        if (corr_e):
          count_arts_adult_gsart += 1

      #print(corr_e.detach().item(), pred_e.detach().item(), trg_e.detach().item() )
    if(n_batches%10 == 0):
      print("Batch", n_batches) 

    if end_of_epoch:
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(float(corr_all) / float(cnt_all)))
      print("tpr student teen: ", float(count_student_teen_gstu)/float(count_student_teen)) 
      print("tpr student adult: ", float(count_student_adult_gstu)/float(count_student_adult))   
      print(float(count_student_adult_gstu)/float(count_student_adult) -  float(count_student_teen_gstu)/float(count_student_teen))
      
      print("tpr arts teen: ", float(count_arts_teen_gsart)/float(count_arts_teen)) 
      print("tpr arts adult: ", float(count_arts_adult_gsart)/float(count_arts_adult))   
      print(float(count_arts_adult_gsart)/float(count_arts_adult) -  float(count_arts_teen_gsart)/float(count_arts_teen))
      print(count_arts_teen,count_arts_adult,count_student_adult,  count_student_teen)
      print(count_arts_adult_gsart,count_arts_teen_gsart, count_student_adult_gstu , count_student_teen_gstu)
      total_loss += valid_loss / n_batches
      total_acc += valid_acc / n_batches
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1

      break
  return (float(corr_all) / float(cnt_all)), total_loss





def eval(model, data, crit, step, hparams):
  print("Eval at step {0}. valid_batch_size={1}".format(step, args.valid_batch_size))
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  cnt = 0
  while True:
    x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, y_neg, batch_size, end_of_epoch, _ = data.next_dev(dev_batch_size=hparams.batch_size)
    # clear GPU memory
    #print(x)
    #gc.collect() #TODO
    # next batch
    #print("here 1")
    logits = model.forward(
      x, x_mask, x_len, step=step)

    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y.view(-1)

    val_loss = crit(logits, labels)
    _, preds = torch.max(logits, dim=1)
 
    val_acc = torch.eq(preds, labels).int().sum()
    #print(labels)
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()

    if end_of_epoch:
      print("val_step={0:<6d}".format(step))
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      total_loss += valid_loss
      total_acc += valid_acc
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
      break
    
  return total_acc / file_count, total_loss

def train():
  if args.load_model and (not args.reset_hparams):
    print("load hparams..")
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.load_model = args.load_model
    hparams.n_train_steps = args.n_train_steps

    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    #optim = torch.optim.Adam(trainable_params, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=hparams.l2_reg)
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_val_ppl, best_val_bleu, cur_attempt, lr = torch.load(extra_file_name)
  else:
    hparams = HParams(**vars(args))

  print("building model...")
  if args.load_model:
    data = DataUtil(hparams=hparams)
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    num_params = count_params(trainable_params)
    print("Model has {0} params".format(num_params))

    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    #optim = torch.optim.Adam(trainable_params, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9)
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_loss, best_acc, cur_attempt, lr = torch.load(extra_file_name)
  else:
    data = DataUtil(hparams=hparams)
    if hparams.classifer == "cnn":
        model = CNNClassify(hparams)
    else:
        model = BiLSTMClassify(hparams)
    if args.cuda:
      model = model.cuda()
    #if args.init_type == "uniform":
    #  print("initialize uniform with range {}".format(args.init_range))
    #  for p in model.parameters():
    #    p.data.uniform_(-args.init_range, args.init_range)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    num_params = count_params(trainable_params)
    print("Model has {0} params".format(num_params))

    optim = torch.optim.Adam(trainable_params, lr=hparams.lr)
    step = 0
    best_loss = None
    best_acc = None
    cur_attempt = 0
    lr = hparams.lr

  #crit = nn.CrossEntropyLoss(reduction='none')
  crit = nn.CrossEntropyLoss(reduce=False)

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  total_loss, total_batch, acc = 0, 0, 0
  model.train()
  epoch = 0
  while True:
    x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, y_sampled, y_sampled_mask, y_sampled_count, y_sampled_len, y_pos_emb_idxs, batch_size,  eop = data.next_train()
    step += 1
    #print(x_train)
    #print(x_mask)
    logits = model.forward(x_train, x_mask, x_len, step=step)
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y_train.view(-1)

    tr_loss = crit(logits, labels)
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()

    acc += val_acc.item()
    tr_loss = tr_loss.sum()
    total_loss += tr_loss.item()
    total_batch += batch_size

    tr_loss.div_(batch_size)
    tr_loss.backward()
    grad_norm = grad_clip(trainable_params, grad_bound=args.clip_grad)
    optim.step()
    optim.zero_grad()
    if eop: epoch += 1
    if step % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format((step) / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " loss={0:<7.2f}".format(total_loss)
      log_string += " acc={0:<5.4f}".format(acc / total_batch)
      log_string += " |g|={0:<5.2f}".format(grad_norm)


      log_string += " wpm(k)={0:<5.2f}".format(total_batch / (1000 * elapsed))
      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)
      acc, total_loss, total_batch = 0, 0, 0
      log_start_time = time.time()

    if step % args.eval_every == 0:
      model.eval()
      cur_acc, cur_loss = eval(model, data, crit, step, hparams)
      if not best_acc or best_acc < cur_acc:
        best_loss, best_acc = cur_loss, cur_acc
        cur_attempt = 0
        save_checkpoint([step, best_loss, best_acc, cur_attempt, lr], model, optim, hparams, args.output_dir)
      else:
        if args.lr_dec:
          lr = lr * args.lr_dec
          set_lr(optim, lr)

        cur_attempt += 1
        if args.patience and cur_attempt > args.patience: break
      model.train()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="classify")

  parser.add_argument("--dataset", type=str, help="dataset name, mainly for naming purpose")

  parser.add_argument("--always_save", action="store_true", help="always_save")
  parser.add_argument("--id_init_sep", action="store_true", help="init identity matrix")
  parser.add_argument("--id_scale", type=float, default=0.01, help="[mlp|dot_prod|linear]")

  parser.add_argument("--semb", type=str, default=None, help="[mlp|dot_prod|linear]")
  parser.add_argument("--dec_semb", action="store_true", help="load an existing model")
  parser.add_argument("--query_base", action="store_true", help="load an existing model")
  parser.add_argument("--semb_vsize", type=int, default=None, help="how many steps to write log")
  parser.add_argument("--lan_code_rl", action="store_true", help="whether to set all unk words of rl to a reserved id")
  parser.add_argument("--sample_rl", action="store_true", help="whether to set all unk words of rl to a reserved id")
  parser.add_argument("--sep_char_proj", action="store_true", help="whether to have separate matrix for projecting char embedding")
  parser.add_argument("--residue", action="store_true", help="whether to set all unk words of rl to a reserved id")
  parser.add_argument("--layer_norm", action="store_true", help="whether to set all unk words of rl to a reserved id")
  parser.add_argument("--src_no_char", action="store_true", help="load an existing model")
  parser.add_argument("--trg_no_char", action="store_true", help="load an existing model")
  parser.add_argument("--char_gate", action="store_true", help="load an existing model")
  parser.add_argument("--shuffle_train", action="store_true", help="load an existing model")
  parser.add_argument("--ordered_char_dict", action="store_true", help="load an existing model")
  parser.add_argument("--out_c_list", type=str, default=None, help="list of output channels for char cnn emb")
  parser.add_argument("--k_list", type=str, default=None, help="list of kernel size for char cnn emb")
  parser.add_argument("--highway", action="store_true", help="load an existing model")
  parser.add_argument("--n", type=int, default=4, help="ngram n")
  parser.add_argument("--single_n", action="store_true", help="ngram n")
  parser.add_argument("--bpe_ngram", action="store_true", help="bpe ngram")
  parser.add_argument("--uni", action="store_true", help="Gu Universal NMT")
  parser.add_argument("--pretrained_src_emb_list", type=str, default=None, help="ngram n")
  parser.add_argument("--pretrained_trg_emb", type=str, default=None, help="ngram n")

  parser.add_argument("--load_model", action="store_true", help="load an existing model")
  parser.add_argument("--reset_output_dir", action="store_true", help="delete output directory if it exists")
  parser.add_argument("--output_dir", type=str, default="", help="path to output directory")
  parser.add_argument("--log_every", type=int, default=50, help="how many steps to write log")
  parser.add_argument("--eval_every", type=int, default=500, help="how many steps to compute valid ppl")
  parser.add_argument("--clean_mem_every", type=int, default=10, help="how many steps to clean memory")
  parser.add_argument("--eval_bleu", action="store_true", help="if calculate BLEU score for dev set")
  parser.add_argument("--beam_size", type=int, default=5, help="beam size for dev BLEU")
  parser.add_argument("--poly_norm_m", type=float, default=1, help="beam size for dev BLEU")
  parser.add_argument("--ppl_thresh", type=float, default=20, help="beam size for dev BLEU")
  parser.add_argument("--max_trans_len", type=int, default=300, help="beam size for dev BLEU")
  parser.add_argument("--merge_bpe", action="store_true", help="if calculate BLEU score for dev set")
  parser.add_argument("--dev_zero", action="store_true", help="if eval at step 0")

  parser.add_argument("--cuda", action="store_true", help="GPU or not")
  parser.add_argument("--decode", action="store_true", help="whether to decode only")

  parser.add_argument("--max_len", type=int, default=10000, help="maximum len considered on the target side")
  parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")

  parser.add_argument("--d_word_vec", type=int, default=288, help="size of word and positional embeddings")
  parser.add_argument("--d_char_vec", type=int, default=None, help="size of word and positional embeddings")
  parser.add_argument("--d_model", type=int, default=288, help="size of hidden states")
  parser.add_argument("--d_inner", type=int, default=512, help="hidden dim of position-wise ff")
  parser.add_argument("--n_layers", type=int, default=1, help="number of lstm layers")
  parser.add_argument("--n_heads", type=int, default=3, help="number of attention heads")
  parser.add_argument("--d_k", type=int, default=64, help="size of attention head")
  parser.add_argument("--d_v", type=int, default=64, help="size of attention head")
  parser.add_argument("--pos_emb_size", type=int, default=None, help="size of trainable pos emb")

  parser.add_argument("--train_src_file", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/test.txt', help="source train file")
  parser.add_argument("--train_trg_file", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/test.attr', help="target train file")
  parser.add_argument("--dev_src_file", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/test.txt', help="source valid file")
  parser.add_argument("--dev_trg_file", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/test.attr', help="target valid file")
  parser.add_argument("--dev_trg_ref", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/test.txt', help="target valid file for reference")
  parser.add_argument("--src_vocab", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/text.vocab', help="source vocab file")
  parser.add_argument("--trg_vocab", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/shakespeare/attr.vocab', help="target vocab file")
  parser.add_argument("--test_src_file", type=str, default=None, help="source test file")
  parser.add_argument("--test_trg_file", type=str, default=None, help="target test file")
  parser.add_argument("--src_char_vocab_from", type=str, default=None, help="source char vocab file")
  parser.add_argument("--src_char_vocab_size", type=str, default=None, help="source char vocab file")
  parser.add_argument("--trg_char_vocab_from", type=str, default=None, help="source char vocab file")
  parser.add_argument("--trg_char_vocab_size", type=str, default=None, help="source char vocab file")
  parser.add_argument("--src_vocab_size", type=int, default=None, help="src vocab size")
  parser.add_argument("--trg_vocab_size", type=int, default=None, help="trg vocab size")

  parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
  parser.add_argument("--valid_batch_size", type=int, default=20, help="batch_size")
  parser.add_argument("--batcher", type=str, default="sent", help="sent|word. Batch either by number of words or number of sentences")
  parser.add_argument("--n_train_steps", type=int, default=100000, help="n_train_steps")
  parser.add_argument("--n_train_epochs", type=int, default=0, help="n_train_epochs")
  parser.add_argument("--dropout", type=float, default=0., help="probability of dropping")
  parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
  parser.add_argument("--lr_dec", type=float, default=0.5, help="learning rate decay")
  parser.add_argument("--lr_min", type=float, default=0.0001, help="min learning rate")
  parser.add_argument("--lr_max", type=float, default=0.001, help="max learning rate")
  parser.add_argument("--lr_dec_steps", type=int, default=0, help="cosine delay: learning rate decay steps")

  parser.add_argument("--n_warm_ups", type=int, default=0, help="lr warm up steps")
  parser.add_argument("--lr_schedule", action="store_true", help="whether to use transformer lr schedule")
  parser.add_argument("--clip_grad", type=float, default=5., help="gradient clipping")
  parser.add_argument("--l2_reg", type=float, default=0., help="L2 regularization")
  parser.add_argument("--patience", type=int, default=-1, help="patience")
  parser.add_argument("--eval_end_epoch", action="store_true", help="whether to reload the hparams")

  parser.add_argument("--seed", type=int, default=0, help="random seed")

  parser.add_argument("--init_range", type=float, default=0.1, help="L2 init range")
  parser.add_argument("--init_type", type=str, default="uniform", help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

  parser.add_argument("--share_emb_softmax", action="store_true", help="weight tieing")
  parser.add_argument("--label_smoothing", type=float, default=None, help="label smooth")
  parser.add_argument("--reset_hparams", action="store_true", help="whether to reload the hparams")

  parser.add_argument("--char_ngram_n", type=int, default=0, help="use char_ngram embedding")
  parser.add_argument("--max_char_vocab_size", type=int, default=None, help="char vocab size")

  parser.add_argument("--char_input", type=str, default=None, help="[sum|cnn]")
  parser.add_argument("--char_comb", type=str, default="add", help="[cat|add]")

  parser.add_argument("--char_temp", type=float, default=None, help="temperature to combine word and char emb")

  parser.add_argument("--pretrained_model", type=str, default=None, help="location of pretrained model")

  parser.add_argument("--src_char_only", action="store_true", help="only use char emb on src")
  parser.add_argument("--trg_char_only", action="store_true", help="only use char emb on trg")

  parser.add_argument("--model_type", type=str, default="seq2seq", help="[seq2seq|transformer]")
  parser.add_argument("--share_emb_and_softmax", action="store_true", help="only use char emb on trg")
  parser.add_argument("--transformer_wdrop", action="store_true", help="whether to drop out word embedding of transformer")
  parser.add_argument("--transformer_relative_pos", action="store_true", help="whether to use relative positional encoding of transformer")
  parser.add_argument("--relative_pos_c", action="store_true", help="whether to use relative positional encoding of transformer")
  parser.add_argument("--relative_pos_d", action="store_true", help="whether to use relative positional encoding of transformer")
  parser.add_argument("--update_batch", type=int, default="1", help="for how many batches to call backward and optimizer update")
  parser.add_argument("--layernorm_eps", type=float, default=1e-9, help="layernorm eps")

  # noise parameters
  parser.add_argument("--word_blank", type=float, default=0.2, help="blank words probability")
  parser.add_argument("--word_dropout", type=float, default=0.2, help="drop words probability")
  parser.add_argument("--word_shuffle", type=float, default=1.5, help="shuffle sentence strength")

  # balance training objective
  parser.add_argument("--anneal_epoch", type=int, default=1,
      help="decrease the weight of autoencoding loss from 1.0 to 0.0 in the first anneal_iter epoch")

  # sampling parameters
  parser.add_argument("--temperature", type=float, default=1., help="softmax temperature during training, a small value approx greedy decoding")
  parser.add_argument("--gumbel_softmax", action="store_true", help="use gumbel softmax in back-translation")

  parser.add_argument("--reconstruct", action="store_true", help="whether perform reconstruction or transfer when validating bleu")
  parser.add_argument("--negate", action="store_true", help="whether negate the labels when evaluating")
  parser.add_argument("--classifer", type=str, choices=["cnn", "lstm"])
  parser.add_argument("--run_classification_eval", action='store_true')
  parser.add_argument("--classifier_dir", type=str, default='/home/user/dir.projects/sent_analysis/deep-latent-sequence-model/pretrained_classifer/shakespeare')
  parser.add_argument("--file_dir", type=str, default='/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/shakespeare/he_etal')


  args = parser.parse_args()

  if args.run_classification_eval:
    
    device = torch.device("cuda" if args.cuda else "cpu")
    classifier_file_name = os.path.join(args.classifier_dir, "model.pt")
    classifier = torch.load(classifier_file_name).to(device)
    classifier.eval()
    hparams = HParams(**vars(args))
    hparams.noise_flag = True
    data = DataUtil(hparams=hparams)
    valid_hyp_file = f'{args.file_dir}/opt_samples.txt'
    
    cur_acc, cur_loss = test(classifier, data, hparams, valid_hyp_file, hparams.dev_trg_file, dir_out=args.file_dir, negate=True)
    print("classifier_acc={}".format(cur_acc))
