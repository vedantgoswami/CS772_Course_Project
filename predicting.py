from transformers import AutoModelForTokenClassification,AutoTokenizer
from transformers import pipeline
from datasets import *
from modelUtils import tokenize_and_align_labels
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")

#model to the GPU
model_fine_tuned.to(device)

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

dataset_dict = DatasetDict.load_from_disk("Dataset/dataset.json")

tokenized_datasets = tokenize_and_align_labels(dataset_dict['test'])
# print(type(tokenized_datasets))
# print(tokenized_datasets['labels'][0])
# print(tokenizer.convert_ids_to_tokens(tokenized_datasets["input_ids"][0]))

count=0
actual_word_Count=0
tokenized_word_Count=0

predicted_label_list = []
actual_label_list = []
for sentence in dataset_dict['test']['tokens'] :
    actual_word_Count+=len(sentence)
    example = " ".join(sentence)
    ner_results = nlp(example)
    pr = []
    for ele in ner_results:
        pr.append(int(ele['entity'][-1]))
    tokenized_word_Count+=len(pr)
    tr= tokenized_datasets['labels'][count]
    actual_label_list.append(tr)
    predicted_label_list.append(pr)
    if count%50 == 0 :
        print(len(sentence))
        print(len(pr))
        print(f'At line {count}')
    count+=1

print("Total Data Processed: ",count)
print("Actual Word Count: ",actual_word_Count)
print("Tokenized Word Count: ",tokenized_word_Count)

# Write predicted labels to predicted.txt
with open('predicted.txt', 'w+') as predicted_file:
    for labels in predicted_label_list:
        predicted_file.write(' '.join(map(str, labels)) + '\n')

# Write actual labels to actual.txt
with open('actual.txt', 'w+') as actual_file:
    for labels in actual_label_list:
        actual_file.write(' '.join(map(str, labels)) + '\n')


