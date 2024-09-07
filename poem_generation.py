import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import sys
import random
from functions import clean,embed,prep
from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from warnings import filterwarnings
filterwarnings("ignore")

clean_corpus_path = sys.argv[1]
word_limit = int(sys.argv[2])
model_path = sys.argv[3]
history_path = sys.argv[4]
training_plot_path = sys.argv[5]

## Loading Data
print("Loading Data")

db = pd.read_csv(f'{clean_corpus_path}')

hehe = db.sample(frac = 1)
hehe["len"] = hehe.text.apply(lambda x : len(x.split()))
hehe = hehe[hehe.len<word_limit]


## Tokenizing Data
print("Tokenizing Data")
class Tokenizer:

    def __init__(self):
        self.word_index = {}
        self.word_count = {}
        self.index_word = {}
        self.oov_token = '<oov>'
        self.vocab_size = None
        self.index = 1

    def fit(self,texts):

        for text in texts:
            text = text.split()
            for word in text:
                if word in self.word_index:
                    self.word_count[word]+=1
                else:
                    self.word_index[word] = self.index
                    self.index += 1
                    self.word_count[word] = 1

        self.index_word = {self.word_index[word] : word for word in self.word_index}

        self.vocab_size = max(self.index_word.keys()) + 1

    def texts_to_sequences(self,texts):

        out = []

        for text in texts:
            text_out = []
            text = text.split()
            for word in text:
                try:
                    text_out += [self.word_index[word]]
                except:
                    text_out += [self.word_index[self.oov_token]]
            out += [text_out]

        return out

    def sequences_to_texts(self,seqs):

        out = []

        for seq in seqs:
            seq_out = []
            for index in seq:
                seq_out += [self.index_word[index]]
            out += [' '.join(seq_out)]

        return out

tk = Tokenizer()
tk.fit(hehe.text)
hehe["seq"] = tk.texts_to_sequences(hehe.text)

print(f"Vocabulary Size: {tk.vocab_size}")

with open("tokenizer.bin","wb") as f:
    pickle.dump(tk,f)

## Creating N-Grams for Training
print("Creating N-Grams for Training")

ngrams = []
types = []
labels = []
for index,row in hehe.iterrows():
    for i in range(2,len(row.seq)):
        ngrams += [row.seq[:i-1]]
        labels += [row.seq[i]]
        types += [row.type]

ngrams = pad_sequences(ngrams,maxlen=word_limit)
types = np.array(types)
labels = to_categorical(labels,tk.vocab_size,dtype=np.int8)

## Building Model
print("Building Model")
ngram_input = Input(shape=(word_limit,))
theme_input = Input(shape=(1,))

embedding_layer = Embedding(input_dim=tk.vocab_size, output_dim=128)(ngram_input)

lstm1= LSTM(256)(embedding_layer)

concatenated = concatenate([lstm1, theme_input])

dense1 = Dense(512, activation='relu')(concatenated)

dropout1 = Dropout(0.3)(dense1)

output = Dense(tk.vocab_size, activation='softmax')(dropout1)

model = Model(inputs=[ngram_input, theme_input], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

from keras.callbacks import Callback

seeds = ["roses are red, violets are blue","the sky is blue", "the king and queen", "the world is happy"]

class SanityCheck(Callback):
    def on_epoch_end(self, epoch, logs=None):
        len = 50
        seed = random.choice(seeds)
        theme = random.randint(1,3)
        
        while len > 0:

            input = tk.texts_to_sequences([seed])
            input = pad_sequences(input,word_limit)

            theme = np.array([theme])

            seed += " "
            seed += tk.index_word[np.argmax(self.model.predict([input,theme]))]

            len -= 1

        print(seed)


## Traning Model

hist = model.fit([ngrams,types],labels,epochs = 50,verbose=2)

model.save(f"{model_path}")

with open(f"{history_path}","w") as f:
    json.dump(hist.history,f) 

fig,ax = plt.subplots()

ax.plot(np.arange(50),hist.history["accuracy"])
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")

fig.savefig(f"{training_plot_path}",dpi=800)