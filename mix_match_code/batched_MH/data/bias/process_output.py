
from csv import reader

dict={'pos':1,'neg':0,'equal':2}

file_dir=  '/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/bias/saap_noboost'

#'/home/user/dir.projects/sent_analysis/sent_anlys/batched_MH/output_samples/bias/saap'
name='test'

def convert_csv(file_dir, name):
    #file_name=f'{file_dir}/{name}'
    file_name =f'{file_dir}/joint-none-test'
    
    list_pair = []
    with open(file_name+".csv", 'r') as read_obj, open(f'{file_dir}/{name}.txt','w') as text_f , open(f'{file_dir}/{name}.attr', 'w') as attr_f, open(f'{file_dir}/{name}_ref.txt','w') as text_ref_f  :
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                text=row[10]
                text_ref = row[1]
                descat = row[7]
                oricat = row[3]
                attr=dict[row[3]]
                if oricat == 'equal':
                    continue
                if oricat == descat or descat == 'equal':
                    continue
                list_pair.append((int(row[0]),text))
                text_f.write(text+"\n")
                text_ref_f.write(text_ref+"\n")
                attr_f.write(str(attr)+"\n")
                text_f.flush()
                attr_f.flush()
                
                
                
    list_pair.sort(key=lambda y: y[0])
    
    with  open(f'{file_dir}/{name}_sorted.txt','w') as text_f:
        for pair in list_pair:
            text_f.write(pair[1]+"\n")
            text_f.flush()

convert_csv(file_dir,name)