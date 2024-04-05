from transformers import pipeline,AutoTokenizer


pipe = pipeline("fill-mask", model="l3cube-pune/hindi-bert-v2")
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hindi-bert-v2")

def tokenize_and_align_labels(examples, label_all_tokens=True): 
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Parameters:
    examples (dict): A dictionary containing the tokens and the corresponding NER tags.
                     - "tokens": list of words in a sentence.
                     - "ner_tags": list of corresponding entity tags for each word.
                     
    label_all_tokens (bool): A flag to indicate whether all tokens should have labels. 
                             If False, only the first token of a word will have a label, 
                             the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token. 
        previous_word_idx = None 
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None 
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids: 
            if word_idx is None: 
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token                 
                label_ids.append(label[word_idx]) 
            else: 
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                # mask the subword representations after the first subword
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs
    
