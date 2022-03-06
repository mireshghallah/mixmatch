
from csv import reader

dict={'pos':1,'neg':0,'equal':2}

file_dir='./'
name='test'

def convert_csv(file_dir, name):
    file_name=f'{file_dir}/{name}'
    
    
    with open(file_name+".csv", 'r') as read_obj, open(f'{file_dir}/{name}_mask.txt','w') as text_f :
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                text=row[2]
                text = text.replace('<VERB>','[MASK]')
                attr=dict[row[3]]
                if attr == 2:
                    continue
                text_f.write(text+"\n")
                text_f.flush()
                

convert_csv(file_dir,name)