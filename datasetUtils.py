import json
from datasets import DatasetDict, Dataset

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

file_path1 = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/AiSight/train.json"
file_path2 = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/AiSight/test.json"
file_path3 = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/AiSight/validation.json"

train_data = load_json(file_path1)
test_data = load_json(file_path2)
valid_data = load_json(file_path3)

X_train = []
Y_train = []

X_temp = []
Y_temp = []
for d in train_data:
    X_temp.append(d['tokens'])
    Y_temp.append(d['ner_tags'])

count_all_O=0

for i,sentence in enumerate(Y_temp):
    if(sum(sentence)==len(sentence)*6):
        count_all_O+=1
    else:
        X_train.append(X_temp[i])
        Y_train.append(sentence)

X_test = []
Y_test = []
for d in test_data:
    X_test.append(d['tokens'])
    Y_test.append(d['ner_tags'])

X_valid = []
Y_valid = []
for d in valid_data:
    X_valid.append(d['tokens'])
    Y_valid.append(d['ner_tags'])

# Combine X and Y into a single list of dictionaries
train_data = [{'tokens': x, 'ner_tags': y} for x, y in zip(X_train, Y_train)]
print(len(train_data))

# Combine X and Y into a single list of dictionaries
test_data = [{'tokens': x, 'ner_tags': y} for x, y in zip(X_test, Y_test)]
print(len(test_data))

# Combine X and Y into a single list of dictionaries
val_data = [{'tokens': x, 'ner_tags': y} for x, y in zip(X_valid, Y_valid)]
print(len(val_data))

# Splitting the data into train (60%), validation (20%), and test (20%)
# train_size = int(0.6 * len(data))
# val_size = int(0.2 * len(data))
# test_size = len(data) - train_size - val_size

# # Splitting the data into train, validation, and test sets
# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]

train_data_dict= {
                    'tokens': [],
                    'ner_tags': []
}

for ele in train_data:
    train_data_dict['tokens'].append(ele['tokens'])
    train_data_dict['ner_tags'].append(ele['ner_tags'])

val_data_dict= {
                    'tokens': [],
                    'ner_tags': []

}
for ele in val_data:
    val_data_dict['tokens'].append(ele['tokens'])
    val_data_dict['ner_tags'].append(ele['ner_tags'])
    
test_data_dict= {
                    'tokens': [],
                    'ner_tags': []

}
for ele in test_data:
    test_data_dict['tokens'].append(ele['tokens'])
    test_data_dict['ner_tags'].append(ele['ner_tags'])
    
# Create DatasetDict with train, validation, and test datasets
dataset_dict = DatasetDict({
    'train': Dataset.from_dict(train_data_dict),
    'validation': Dataset.from_dict(val_data_dict),
    'test': Dataset.from_dict(test_data_dict)
})

dataset_dict.save_to_disk("Dataset/dataset.json")