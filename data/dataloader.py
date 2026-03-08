import os
import re

DATA_DIR = 'data/raw/20_newsgroups'

def clean_data(text):

    text = re.sub(r'^[\w-]+: .+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    return text

def load_data(DATA_DIR = "data/raw/20_newsgroups"):
    doc = []
    labels = []
    for cat in os.listdir(DATA_DIR):
        cat_path = os.path.join(DATA_DIR, cat)
        if os.path.isdir(cat_path):
            for file in os.listdir(cat_path):
                file_path = os.path.join(cat_path, file)
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                    cleaned_text = clean_data(text)
                    doc.append(cleaned_text)
                    labels.append(cat)
    
    return doc, labels