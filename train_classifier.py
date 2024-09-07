import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import words
from tqdm import tqdm
import sys
import pickle
from functions import clean,prep,embed 

model_path = sys.argv[1]

## Loading Data
db = pd.read_csv("datasets/all.csv")
db = db[['content','type']]

print("Data Loaded")

db = clean(db,'content')
xtrain, xtest, ytrain, ytest = prep(db,'content','type')

## Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

classifier = LogisticRegression(max_iter=500,multi_class="multinomial",class_weight='balanced')
classifier.fit(xtrain,ytrain)

print(classification_report(ytest,classifier.predict(xtest)))

print("Classifier Trained")

with open(f"{model_path}",'wb') as f:
    pickle.dump(classifier,f)

print(f"Classifier Model Saved to {model_path}")