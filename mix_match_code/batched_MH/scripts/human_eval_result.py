import csv
from collections import defaultdict
import pandas as pd

res_file = "./sample_generations/human_evals_fudge_pplm/human_pplm/len12_turk_results.csv"

corr_dict = defaultdict(int)
cnt_dict = defaultdict(int)

id_line = -7


df = pd.read_csv(res_file)

for index, row in df.iterrows():
    cnt_dict[row['Input.ID']] += 1

    if row['Input.target'].lower() == row['Answer.Fluency .label'].lower().replace(' ',''):
        corr_dict[row['Input.ID']] += 1
    
    
cnt = 0
corr = 0   

for key1 in cnt_dict.keys():
    cnt += 1
    if corr_dict[key1]>1:
        corr+=1
        
print("acc is", corr/cnt)
    
    
res_file = "./sample_generations/human_evals_fudge_pplm/human_pplm/len20_turk_results.csv"

corr_dict = defaultdict(int)
cnt_dict = defaultdict(int)

id_line = -7


df = pd.read_csv(res_file)

for index, row in df.iterrows():
    cnt_dict[row['Input.ID']] += 1

    if row['Input.target'].lower() == row['Answer.Fluency .label'].lower().replace(' ',''):
        corr_dict[row['Input.ID']] += 1
    
    
cnt = 0
corr = 0   

for key1 in cnt_dict.keys():
    cnt += 1
    if corr_dict[key1]>1:
        corr+=1
        
print("acc is", corr/cnt)


res_file = "./sample_generations/human_evals_fudge_pplm/human_pplm/len50_turk_results.csv"

corr_dict = defaultdict(int)
cnt_dict = defaultdict(int)

id_line = -7


df = pd.read_csv(res_file)

for index, row in df.iterrows():
    cnt_dict[row['Input.ID']] += 1

    if row['Input.target'].lower() == row['Answer.Fluency .label'].lower().replace(' ',''):
        corr_dict[row['Input.ID']] += 1
    
    
cnt = 0
corr = 0   

for key1 in cnt_dict.keys():
    cnt += 1
    if corr_dict[key1]>1:
        corr+=1
        
print("acc is", corr/cnt)