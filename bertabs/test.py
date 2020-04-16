
import csv
filepath ="/home/mhxia/whou/workspace/code_repo/space12-ner-allennlp/data/AIAA_with_text.csv"

title_length =[]
abstract_length= []
with open(filepath ,encoding='utf-8') as f:
    reader = csv.DictReader(f)
    title = reader.fieldnames
    for index, row in enumerate(reader):
        title_length.append(len(row['con_title'].split())) 
        if len(row['con_title'].split())< 2:
            print(row)
        abstract_length.append(len(row['con_abstract'].split()))

print(min(title_length), max(title_length), sum(title_length)//len(title_length))
print(min(abstract_length), max(abstract_length), sum(abstract_length)//len(abstract_length))