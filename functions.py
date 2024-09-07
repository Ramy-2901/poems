import pandas as pd
import numpy as np
from string import punctuation
from tqdm import tqdm
import pickle
import os

import nltk
nltk.download("words")
from nltk.corpus import words


## Cleaning Data

def clean(db,col):

    content = db[col]

    content = content.apply(lambda x : x.lower()) # Lower sentences
    content = content.apply(lambda x : ''.join([i for i in x if i == " " or i not in punctuation])) # De-punctuating
    content = content.apply(lambda x : ''.join([i if i == " " or i.isalnum() else " " for i in x])) # Removing Symbols
    content = content.apply(lambda x : x.split()) # Sentences to words

    # Removing non-words
    vocab = []
    wordlistCheck = {}
    wordlist = [word.lower() for word in words.words()]

    for i in tqdm(range(len(content))):
        out = []
        for word in content[i]:
            try:
                if wordlistCheck[word] == True:
                    out += [word]
            except:
                if word in wordlist:
                    wordlistCheck[word] = True
                    out += [word]
                    vocab += [word]
                else:
                    wordlistCheck[word] = False
        content[i] = out

    db[col] = content

    print("Data Cleaned")

    return db

## Create Embeddings

def embed(db,col):

    if "glove.bin" not in os.listdir():
        import gensim.downloader
        print("Loading GloVe Model")
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        glove_vectors.save("glove.d2v")
    else:
        from gensim.models import KeyedVectors
        glove_vectors = KeyedVectors.load("glove.d2v")

    print("GloVe Model Loaded")
    mean_vectors = db[col].apply(lambda x : list(glove_vectors.get_mean_vector(x)))

    print("Embeddings Created")

    return mean_vectors

## Prepping Data for Classifier

def prep(db,xcol,ycol):

    mean_vectors = embed(db,xcol)

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    encoder = LabelEncoder()
    encoder.fit(db[ycol])
    db.type = encoder.transform(db[ycol])
    mappings = dict(enumerate(encoder.classes_))

    print(mappings)

    with open("mappings",'wb') as f:
        pickle.dump(mappings,f)

    x = mean_vectors.to_list()
    y = db.type.to_list()

    x = np.array(x)
    y = np.array(y)

    return train_test_split(x,y,test_size=0.1)

    print("Data Prepped")
