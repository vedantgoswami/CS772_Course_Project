import streamlit as st
from transformers import AutoModelForTokenClassification,AutoTokenizer
from transformers import pipeline
from datasets import *
from modelUtils import tokenize_and_align_labels
import matplotlib.pyplot as plt
import numpy as np
import torch

label_to_color = {
    'LABEL_0': '#CC0000',   # Darker shade of Red for LABEL_0
    'LABEL_1': '#CC4D33',   # Darker shade of Orange for LABEL_1
    'LABEL_2': '#1FBF26',   # Darker shade of Green for LABEL_2
    'LABEL_3': '#ff4d4d',   # Lighter shade of Red for LABEL_3
    'LABEL_4': '#db8270',   # Lighter shade of Orange for LABEL_4
    'LABEL_5': '#91ed96',   # Lighter shade of Green for LABEL_5
    'LABEL_6': '#CCCCCC',   # Grey
}


class NER():
    def __init__(self, tokenizer_path, model_path) -> None:
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
      self.model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
      self.nlp = pipeline("ner", model=self.model_fine_tuned, tokenizer=self.tokenizer)
    
    def predict(self,sentence):
      return self.nlp(sentence)  


def main():
    st.set_page_config(page_title="Hindi NER", page_icon=":pencil2:", layout="centered")
    
    # Set title
    st.title("Project: Hindi NER")
    st.write("Named Entity Recognition system for the Hindi language.")
    
    # Input fields for sentence
    sentence = st.text_area("Enter the sentence:")
    # pos_list = st.text_area("Enter POS Tags (space-separated):")
    ner_model = NER("Vedantg68/HiNER_CS772", "Vedantg68/HiNER_Tokenizer")
    # Button to submit and trigger the prediction
    if st.button("Submit"):
        # Tokenize the input
        ner_tag = ner_model.predict(sentence)
        
        print(ner_tag)

        colored_words_html = ""
        
        prev_word = ""
        start=0
        end=0
        label=""
        color=""
        for word in ner_tag:
            if word['word'][0]== '#':
                end=word['end']
            else:
                prev_word = sentence[start:end]
                colored_words_html+=f"<div style='background-color: {color}; padding: 5px; border-radius: 5px; margin: 0px 5px 5px 0px; display: inline-block;'>{prev_word}</div>"#f"<span style='background-color: {bg_color}; padding: 5px; border-radius: 5px; margin-right: 2px; margin-top: 10px;font-size: 16px;'>{word}</span> "
                label = word['entity']
                start = word['start']
                end = word['end']
                color = label_to_color[label]
                prev_word = sentence[start:end]
                
        # Color the sentence based on the predicted chunk tags
        # colored_sentence = color_text(sentence, chunk_tag)
        
        # # Display the colored sentence
        # colored_words_html = ""
        # words = sentence.split(" ")
        # for word, tag in zip(words, colored_sentence):
        #     # Determine background color based on chunk tag
        #     bg_color = "#439ee2" if tag == 1 else "#7ee243"
        #     # Wrap the word in a span with background color
        #     colored_words_html += f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; margin: 0px 5px 5px 0px; display: inline-block;'>{word}</div>"#f"<span style='background-color: {bg_color}; padding: 5px; border-radius: 5px; margin-right: 2px; margin-top: 10px;'>{word}</span> "
        
        # Display the colored words inside a box with colored background
        st.markdown(
            f"""
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
            <span style='background-color: #b02346; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>Hindi NER:</strong>
            </span>
            <span style='background-color: #CC0000; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>B_LOC</strong>
            </span>
            <span style='background-color: #CC4D33; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>B_ORG</strong>
            </span>
            <span style='background-color: #1FBF26; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>B_PER</strong>
            </span>
            <span style='background-color: #ff4d4d; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>I_LOC</strong>
            </span>
            <span style='background-color: #db8270; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>I_ORG</strong>
            </span>
            <span style='background-color: #91ed96; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>I_PER</strong>
            </span>
            <span style='background-color: #CCCCCC; padding: 5px; border-radius: 5px; margin-right: 5px;'>
                <strong>O</strong>
            </span>
            <div style='padding: 20px; border: 2px solid #000; border-radius: 5px; margin-top: 10px;'>
                {colored_words_html}
            </div>
        </div>

            """,
            unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()
