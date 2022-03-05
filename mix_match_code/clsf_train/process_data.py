import json
original_txt = "/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/form_em/train.txt"
original_attr = "/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/data/form_em/train.attr"

output_data = "./data/form_em/train.json"

with open(f"{output_data}", "w") as f, open(f"{original_txt}", "r") as data_file, open(f"{original_attr}", "r") as attr_file:
    for txt,attr in zip (data_file,attr_file):
        dict={}
        dict["label"] = [int(attr)]
        dict["text"]  = [txt[:-1]]
        jstr = json.dumps(dict)
        f.write(jstr+'\n')
        f.flush()
