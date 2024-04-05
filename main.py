import datasets
from datasets import *
from transformers import pipeline,AutoTokenizer,AutoModelForTokenClassification,TrainingArguments,Trainer,DataCollatorForTokenClassification
from modelUtils import * 
import warnings
import numpy as np

# Ignore all warnings
warnings.filterwarnings('ignore')




# Load dataset from JSON file
dataset_dict = DatasetDict.load_from_disk("Dataset/dataset.json")
train_data = dataset_dict['train']
validation_data = dataset_dict['validation']
test_data = dataset_dict['test']


tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained("l3cube-pune/hindi-bert-v2",num_labels=7,device_map="cuda:0")

from transformers import TrainingArguments, Trainer 
args = TrainingArguments(
    "test-ner",
    evaluation_strategy = "epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    num_train_epochs=3, 
    weight_decay=0.01, 
) 

data_collator = DataCollatorForTokenClassification(tokenizer) 
metric = datasets.load_metric("seqeval",trust_remote_code=True) 

example = dataset_dict['train'][1]

label_list = {
    '0' : 'B_LOCATION',
    '1' : 'B_ORGANIZATION',
    '2' : 'B_PERSON',
    '3' : 'I_LOCATION',
    '4' : 'I_ORGANIZATION',
    '5' : 'I_PERSON',
    '6' : 'O'
}

labels = [label_list[str(i)] for i in example["ner_tags"]] 

def compute_metrics(eval_preds): 
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds 
    print(pred_logits)
    print(labels)
    pred_logits = np.argmax(pred_logits, axis=2) 
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    
    # We remove all the values where the label is -100
    predictions = [ 
        [label_list[str(eval_preds)] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ]
    print(predictions)
    
    true_labels = [ 
      [label_list[str(l)] for (eval_preds, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
   ] 
    results = metric.compute(predictions=predictions, references=true_labels) 
    return { 
   "precision": results["overall_precision"], 
   "recall": results["overall_recall"], 
   "f1": results["overall_f1"], 
  "accuracy": results["overall_accuracy"], 
  }

tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hindi-bert-v2")

trainer = Trainer( 
    model, 
    args, 
   train_dataset=tokenized_datasets["train"], 
   eval_dataset=tokenized_datasets["validation"], 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics 
)

trainer.train() 
model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")
