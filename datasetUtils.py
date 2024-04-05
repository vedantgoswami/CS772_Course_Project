import json
from datasets import DatasetDict, Dataset

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

file_path = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/AiSight/train.json"
data = load_json(file_path)

X = []
Y = []
for d in data:
    X.append(d['tokens'])
    Y.append(d['ner_tags'])

# Combine X and Y into a single list of dictionaries
data = [{'tokens': x, 'ner_tags': y} for x, y in zip(X, Y)]

# Splitting the data into train (60%), validation (20%), and test (20%)
train_size = int(0.6 * len(data))
val_size = int(0.2 * len(data))
test_size = len(data) - train_size - val_size

# Splitting the data into train, validation, and test sets
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

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