import csv
from collections import defaultdict
import pandas as pd

res_file = "/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/human_pplm/len12.csv"

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
    
    
res_file = "/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/human_pplm/len20.csv"

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


res_file = "/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/human_pplm/len50.csv"

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