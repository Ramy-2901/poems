import pandas as pd
import numpy as np
import pickle
import os
import sys
from functions import clean,embed,prep
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

model_path = sys.argv[1]
clean_corpus_path = sys.argv[2]

## Loading Corpus
dirs = ["./datasets/topics/" + dir_ for dir_ in os.listdir("./datasets/topics/")]

db = []

print("Loading Corpus")

for dir_ in tqdm(dirs):
    for file in os.listdir(dir_):
        try:
            with open(dir_ + "/" + file) as f:
                db += [{"type" : dir_.split("/")[-1], "text" : f.read()}]

        except:
            continue

db = pd.DataFrame.from_dict(db)
db.drop('type',axis=1,inplace=True)


## Preprocessing Corpus and Creating Embeddings for Classification
print("Preprocessing Corpus and Creating Embeddings for Classification")

db_ = clean(db.copy(deep=True),'text')

db_["len"] = db_.text.apply(lambda x : len(x))

db_.drop(db_[db_.len==0].index,inplace=True)
db.drop(db_[db_.len==0].index,inplace=True)

vectors = embed(db_,'text')
vectors = np.array([vector for vector in vectors])


## Loading Classifier and Predicting Themes
print("Loading Classifier and Predicting Themes")

with open(f'{model_path}', 'rb') as f:
    classifier = pickle.load(f)

out = classifier.predict(vectors)

db_["type"] = out

db = db.iloc[db_.index]
db['type'] = db_.type


## Cleaning Corpus for Text Generation
print("Cleaning Corpus for Text Generation")

from string import punctuation

db.text = db.text.apply(lambda x : x.lower()) # Lower sentences
db.text = db.text.apply(lambda x : ''.join([i for i in x if i == " " or i not in punctuation])) # De-punctuating
db.text = db.text.apply(lambda x : x.replace('\n',' , '))
db.text = db.text.apply(lambda x : ''.join([i if i in [" ",","] or i.isalnum() else " " for i in x])) # Removing Symbols

db.to_csv(f'{clean_corpus_path}')

print(f"Clean Corpus Saved to {clean_corpus_path}")