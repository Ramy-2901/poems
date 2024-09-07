import numpy as np
import random
import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.saving import load_model

model = load_model("model.keras")

seed = input("Enter input: ")
theme = int(input(
  """
  1) Love
  2) Mythology and Folklore
  3) Nature
  Choose theme: 
  """
))
word_limit = int(sys.argv[1])

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

with open("tokenizer.bin","rb") as f:
  tk = pickle.load(f)

input = tk.texts_to_sequences([seed])
input = pad_sequences(input,word_limit)

theme = np.array([theme])

len = 50

while len > 0:

  input = tk.texts_to_sequences([seed])
  input = pad_sequences(input,word_limit)

  seed += " "
  seed += tk.index_word[np.argmax(model.predict([input,theme]))]

  len -= 1

print(seed)