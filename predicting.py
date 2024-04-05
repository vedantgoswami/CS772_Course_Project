from transformers import AutoModelForTokenClassification,AutoTokenizer
from transformers import pipeline
from datasets import *

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
example = "बिल गेट्स माइक्रोसॉफ्ट के संस्थापक हैं"

dataset_dict = DatasetDict.load_from_disk("Dataset/dataset.json")
example = " ".join(dataset_dict['test'][1790]['tokens'])
tr= dataset_dict['test'][1790]['ner_tags']
print(example)
ner_results = nlp(example)

for pred,actual in zip(ner_results,tr):
    print(pred,"   ",actual)