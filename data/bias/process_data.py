
from csv import reader

dict={'pos':1,'neg':0,'equal':2}

file_dir='./'
name='test'

def convert_csv(file_dir, name):
    file_name=f'{file_dir}/{name}'
    
    
    with open(file_name+".csv", 'r') as read_obj, open(f'{file_dir}/{name}.txt','w') as text_f , open(f'{file_dir}/{name}.attr', 'w') as attr_f :
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                text=row[1]
                attr=dict[row[3]]
                #if attr == 2:
                #    continue
                text_f.write(text+"\n")
                attr_f.write(str(attr)+"\n")
                text_f.flush()
                attr_f.flush()
                

convert_csv(file_dir,name)