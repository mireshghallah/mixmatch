
import subprocess
import re
import argparse
import os
import math
import csv

parser = argparse.ArgumentParser(description="metrics")

parser.add_argument("--checkpoint_dir_1", type=str)
parser.add_argument("--checkpoint_dir_2", type=str)


parser.add_argument("--data_file", type=str, default="/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/topic/")



dict_topics =   {'0':'computers', '1':'legal', '2':'military', '3':'politics','4':'religion', '5':'science', '6':'space'}

args = parser.parse_args()


test_file_1=args.data_file+'test_1.txt'
attr_file_1=args.data_file+'test_1.attr'

test_file_2= args.data_file+'test_2.txt'
attr_file_2= args.data_file+'test_2.attr'


prefix ='/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/topic/'
input_samples_1= prefix+args.checkpoint_dir_1+'/'+'opt_samples.txt'
input_samples_2= prefix+args.checkpoint_dir_2+'/'+'opt_samples.txt'


output_file= prefix+args.checkpoint_dir_1+'/'+'eval_predictions.log'



with open(test_file_1,'r') as test_f_1, open(test_file_2,'r') as test_f_2, open(attr_file_1,'r') as attr_f_1, open(attr_file_2,'r') as attr_f_2, open(input_samples_1,'r') as input_f_1, open(input_samples_2,'r') as input_f_2, open(output_file,'w') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['category',  'generation'])
        writer.writeheader()

        for line_in1, attr_in1, generation1 in zip (test_f_1,attr_f_1,input_f_1):
            category = dict_topics[attr_in1.strip('\n')]
            generation = generation1.strip('\n')  
            writer.writerow({'category': category,  'generation': generation})
            
        for line_in2, attr_in2, generation2 in zip (test_f_2,attr_f_2,input_f_2):
            category = dict_topics[attr_in2.strip('\n')]
            generation = generation2.strip('\n')  
            writer.writerow({'category': category,  'generation': generation})
            
            
        