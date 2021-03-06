# -*- coding: utf-8 -*-
"""Janta-NLPHack_bert2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JMeZnEj-JHalrVPUATSjdGCEDnyr7K2e
"""

!wget https://www.dropbox.com/sh/kgplwtz9434ihms/AADhgaqy81V6_1wRe8EIEIvLa?dl=0

!ls

from zipfile import ZipFile
file_name="AADhgaqy81V6_1wRe8EIEIvLa?dl=0"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("done")

from zipfile import ZipFile
file_name="train_E52nqFa.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("done")

from zipfile import ZipFile
file_name="test_BppAoe0.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("done")

from zipfile import ZipFile
file_name="test_BppAoe0.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("done")

!ls

import pandas as pd
train=pd.read_csv("train.csv")
overview=pd.read_csv('game_overview.csv')
test=pd.read_csv("test.csv")
submit=pd.read_csv("sample_submission_wgBqZCk.csv")
print("train shape : ",train.shape)
print("test shape : ",test.shape)
print("overview shape : ",overview.shape)

train_df=pd.merge(train,overview,how='left',on='title')
train_df.shape

train_df['tags_new'] = train_df['tags'].apply(lambda x: x[1:-1])

test_df=pd.merge(test,overview,how='left',on='title')
test_df.shape

test_df.head()

test_df['tags_new'] = test_df['tags'].apply(lambda x: x[1:-1])

train_df['text']=train_df['title']+' '+ train_df['user_review']+' '+ train_df['tags_new']+' '+ train_df['overview']
train_df.head()

test_df['text']=test_df['title']+' '+ test_df['user_review']+' '+ test_df['tags_new']+' '+ test_df['overview']
test_df.head()

import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import Word

nltk.download('stopwords')

data=train_df[['text','user_suggestion']]
data.shape

data1=test_df[['text']]
data1.shape

data['char_count'] = data['text'].str.len() ## this also includes spaces
#data[['text','char_count']].tail()
data['char_count'].describe()

data1['char_count'] = data1['text'].str.len() ## this also includes spaces
#data[['text','char_count']].tail()
data1['char_count'].describe()

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['text'].apply(lambda x: avg_word(x))
data[['text','avg_word']].head()

data1['avg_word'] = data1['text'].apply(lambda x: avg_word(x))
data1[['text','avg_word']].head()

data1['text'][0]

stop = stopwords.words('english')

data['stopwords'] = data['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['text','stopwords']].head()

data1['stopwords'] = data1['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data1[['text','stopwords']].head()

data['numerics'] = data['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['text','numerics']].head()

data1['numerics'] = data1['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data1[['text','numerics']].head()



data.describe()

data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data1['text'] = data1['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

data.head()

stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data1['text'] = data1['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data.head()

#####Removing pantuation

data['text'] = data['text'].str.replace('[^\w\s]','')
data1['text'] = data1['text'].str.replace('[^\w\s]','')

nltk.download('wordnet')
from textblob import Word

data['text']  = data['text'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data.head()

data1['text']  = data1['text'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data1.head()

##Remove special characters
import re

data['text']=data['text'].apply(lambda x :re.sub(r'\W+', ' ', x))
data1['text']=data1['text'].apply(lambda x :re.sub(r'\W+', ' ', x))

data.head()

####Remove numbers
data['text']=data['text'].str.replace('\d+', '')
data1['text']=data1['text'].str.replace('\d+', '')
data.head()

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

!pip install sentencepiece

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow_hub as hub
import tokenization

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Commented out IPython magic to ensure Python compatibility.
# %%time
# module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
# bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(data.text.values, tokenizer, max_len=160)
test_input = bert_encode(data1.text.values, tokenizer, max_len=160)
train_labels = data.user_suggestion.values

model = build_model(bert_layer, max_len=160)
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    callbacks=[es], # Early stopping
    epochs=3,
    batch_size=16
)

model.save('model.h5')

test_pred = model.predict(test_input)

test_pred

submit['user_suggestion'] = test_pred.round().astype(int)
submit.to_csv('submit2_bert.csv', index=False)

submit.head()

from google.colab import files
files.download('submit2_bert.csv')