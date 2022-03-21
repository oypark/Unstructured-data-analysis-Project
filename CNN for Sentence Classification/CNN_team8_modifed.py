#!/usr/bin/env python
# coding: utf-8

# # CNN for Sentence Classification 논문 구현
# 
# #### Git - Commit Message Convention
# 
# * 제가 대략적으로 framework을 잡아놓은 것이니, 담당하시는 부분에 수정이 필요하시면 마크다운 양식만 유지한 채 수정해주시면 됩니다!
# * 작은 단위의 작업이 끝날 때 마다 git add, commit, push 해주시면 됩니다! (push하고 슬랙에 공유 부탁드려요! 화이팅!)
# 
# * git commit message는 다음의 양식을 참고해주세요!
#     * 처음으로 코드 완료했을 때 git commit -m "동사 명사"
#     ```ex) git commit -m "Fill and replace NaN values"```
#     
#     * commit 했던 코드를 수정했을 때 git commit -m "Update 수정한 내용"
#     ```ex) git commit -m "Update Word embedding"```
# 
# 
# # 1. 데이터 load 및 EDA



# ## 1) 네이버 영화 리뷰 데이터 불러오기

from tqdm import tqdm

import pandas as pd

train = pd.read_csv("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", sep='\t')    #일단 제걸로 경로를 걸어놨습니다-옥영
test = pd.read_csv("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", sep='\t')

## 결측치 제거
train.dropna(inplace=True)
test.dropna(inplace=True)


# # 2. Data Preprocessing

# ### colab에서 konlpy, mecab 설치하기

# ## 1) Tokenizer
# 
# * 어절 단위
# * 형태소 단위 - Mecab 활용
# * Subword 단위

# 영어, 한글만 포함하고 나머지 제거
import re

def preprocess(text):
  text = re.sub(r"[^A-Za-zㄱ-ㅎㅏ-ㅣ가-힣 ]","", text) 
  return text

train['document'] = train.document.apply(lambda x : preprocess(x))
test['document'] = test.document.apply(lambda x : preprocess(x))

# Mecab으로 형태소 분석, 불용어 제거
from konlpy.tag import Mecab

tokenizer = Mecab()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']   #korean_stopwords 보류

train_tokens = []
test_tokens = []
train_drop_idx = []
test_drop_idx = []

for idx, sentence in tqdm(enumerate(train['document'])):
    tokenized_sentence = tokenizer.morphs(sentence) 
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    if len(stopwords_removed_sentence) != 0:
      train_tokens.append(stopwords_removed_sentence)
    else:
      train_drop_idx.append(idx)

for idx, sentence in tqdm(enumerate(test['document'])):
    tokenized_sentence = tokenizer.morphs(sentence) 
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    if len(stopwords_removed_sentence) != 0:
      test_tokens.append(stopwords_removed_sentence)
    else:
      test_drop_idx.append(idx)


## 길이 0인 리뷰 labels에서 제거

labels_train = train['label']
labels_test = test['label']

labels_train.drop(train_drop_idx, inplace=True)
labels_test.drop(test_drop_idx, inplace=True)


# tokens 최대 길이를 (평균 + 1.5 * 표준편차)로 설정
import numpy as np

train_token_len = [len(tokens) for tokens in train_tokens]
test_token_len = [len(tokens) for tokens in test_tokens]

max_tokens = np.mean(train_token_len) + 1.5 * np.std(train_token_len)
max_length = int(max_tokens)

print('pad_sequences maxlen : ', max_length)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(train_token_len < max_tokens) / len(train_token_len) * 100))


# ## 3) Word Vectorize
# 
# * word embedding : 문장들을 word vector 형태로 변환
#     * 윗단에서 tokenizer output을 '문장' 형태라고 가정하고 코드 작업
#     1. 문장을 토큰으로 쪼갠다
#     2. 쪼개진 토큰을 가장 긴 문장에 맞춰 패딩한다
#     3. 패딩이 마친 토큰들을 word vector로 변환하다
# * oov, padding, truncating 확인

train_sentences = [" ".join(tokens) for tokens in train_tokens]
test_sentences = [" ".join(tokens) for tokens in test_tokens]

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


## vocabulary, sequences 생성 및 padding 함수

def word_vectorize(sentences, max_length, oov_tok='<oov>',trunc_type='post'):

    tokenizer = Tokenizer(oov_token="<oov>")    #Tokenizer 객체 생성
    tokenizer.fit_on_texts(sentences)    #train sentences 입력
    word_index = tokenizer.word_index           #word : index 사전(vocab) 생성
    sequences = tokenizer.texts_to_sequences(sentences)     #시퀀스 생성
    padded = pad_sequences(sequences, maxlen=max_length, padding=trunc_type, truncating=trunc_type)   #padding, truncating

    return word_index, sequences, padded, tokenizer

word_index, sequences, padded, tokeniz = word_vectorize(train_sentences, max_length)
# word_index_test, sequences_test, padded_test = word_vectorize(test_sentences, max_length)
sequences_test = tokeniz.texts_to_sequences(test_sentences)
padded_test = pad_sequences(sequences_test, maxlen=max_length, padding='post', truncating='post')

index_word = {value : key for (key, value) in word_index.items()}    #index : word 사전

## 시퀀스 decoding 함수
def decode_review(text):
  return " ".join([index_word.get(i, '?') for i in text])


# ## 4) FastText pre-trained model 불러오기
# 
# * pre-trained model : https://github.com/Kyubyong/wordvectors
#     * korean fasttext file download 후 **ko.bin**(word vector + model) 파일 로드
#     * gensim.models.fasttest의 load_facebook_model() 메소드는 로컬에서 돌아가지만, colab에서는 지원하지 않음
#     * **colab에서는 gensim.models.fasttest.FastText의 load_fasttext_format() 메소드를 사용해야함!**
# 
# * (참고) facebook에서 제공하는 기본 pre-trained 모델 (성능이 별로 좋지 않다고 함) : https://fasttext.cc/docs/en/crawl-vectors.html

## 단어 벡터 가져오기

import datetime
from gensim.models.keyedvectors import KeyedVectors
print(f"Load fasttext START at {datetime.datetime.now()}")
fasttext = KeyedVectors.load_word2vec_format("/content/drive/MyDrive/2022mulcam/models/ko.vec")
print(f"Load fasttext   END at {datetime.datetime.now()}")

## word vocabulary list 생성
vocabulary = list(word_index.keys())
vocab_size = len(vocabulary)+1     #52170 >> 52171

embedding_matrix = np.zeros((vocab_size, 200))

for i, word in tqdm(enumerate(vocabulary)): 
    if word in fasttext: 
        embedding_vector = fasttext[word] 
        embedding_matrix[i] = embedding_vector
embed_dim = len(embedding_matrix[1])

# ## 5) Valid, Test set 생성
# * valid : test = 0.8 : 0.2

from sklearn.model_selection import train_test_split
valid_padded, test_padded, valid_labels, test_labels = train_test_split(padded_test, labels_test, test_size=0.2, shuffle=True)

labels_train = np.array(labels_train)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)

# # 3. Convolutional Neural Networks Modeling

# ## 1) 기본 Model 생성
# 
# * Check points
# * padding, initializer
# * activation function
# * Dropout
# * Batch Normalization
# * optimizer


import tensorflow as tf
from keras import Model
# from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Concatenate, Layer
# Sentences Classification Model
# tf.config.run_functions_eagerly(True)# debugging용
# class ConvMultiMap(Model):
class ConvMultiMap(Layer):
  def __init__(self, kernel_size, activation='relu', padding='same'):
    super(ConvMultiMap, self).__init__()
    print('convolution map')
    # self.conv1 = Conv1D(filters=100, kernel_size=kernel_size, activation=activation, padding=padding)
    # self.pool1 = MaxPool1D()
    self.conv1 = Conv1D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding)
    self.pool1 = MaxPool1D()
    self.conv2 = Conv1D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)
    self.pool2 = MaxPool1D()
    self.conv3 = Conv1D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding)
    
  def call(self,inputs, training=False):     #default값 파라미터 이름 주의!
    # print('conv layer:', inputs.shape)#?tf.config.run_functions_eagerly(True)
    net = self.conv1(inputs)
    net = self.pool1(net)
    net = self.conv2(net)
    net = self.pool2(net)
    net = self.conv3(net)

    return net


# class Denselayer(Model):
class Denselayer(Layer):
  def __init__(self, activation='relu'):
    print('dense layer')
    super(Denselayer, self).__init__()
    self.dense4 = Dense(units=128, activation=activation)
    self.drop = Dropout(0.5)      #Bernnoulli 분포로 0.5% dropout!
    # self.outputs = Dense(units=1, activation='softmax')
    self.outputs = Dense(units=1, activation='sigmoid')

  def call(self,inputs, training=False):
    # print('dense layer:', inputs.shape)#?tf.config.run_functions_eagerly(True)
    net = self.dense4(inputs)
    net = self.drop(net)
    net = self.outputs(net)

    return net


class ReviewModel(Model):
  def __init__(self, *args, **kwargs):
      super(ReviewModel, self).__init__(name=kwargs['model_name'])
      self.embed = Embedding(kwargs['vocab_size'], kwargs['embed_dim'],
            weights=[kwargs['embedding_matrix']], input_length=kwargs['max_length'], trainable=kwargs['trainable'])   #trainable=False => CNN-static 모델

      self.conv_list = [ConvMultiMap(kernel_size=kernel_size) for kernel_size in [3,4,5]]
      self.denseBlock = Denselayer(kwargs['activation'])

  def call(self, inputs, training=None, mask=None):
      net = self.embed(inputs)
      net = tf.concat([conv(net) for conv in self.conv_list], axis=-1)
      net = self.denseBlock(net)
      return net


# kernel_size = [3, 4, 5]
# conv_blocks = []
# trainable = True
# for size in kernel_size:
#   conv_block = ConvMultiMap(embedding_matrix=embedding_matrix, inputs=padded, kernel_size=size, trainable=trainable)
#   conv_blocks.append(conv_block)

# model = tf.keras.layers.Concatenate(conv_blocks)

# model = Denselayer(model)
kargs={
  'model_name': 'test'
  , 'embedding_matrix': embedding_matrix
  , 'trainable':False
  , 'vocab_size':vocab_size
  , 'embed_dim':embed_dim
  , 'max_length':max_length
  , 'activation':'relu'
}
model = ReviewModel(**kargs)

input_size = (max_length, )
model(Input(shape=input_size))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint, EarlyStopping

## 모델 weights 저장
MODEL_SAVE_FOLDER = '/content/drive/MyDrive/2022mulcam/models'
model_path = f"{MODEL_SAVE_FOLDER}CNN-{{epoch:d}}-{{val_loss:.5f}}-{{val_accuracy:.5f}}.hdf5"

## checkpoint 설정
cb_checkpoint = ModelCheckpoint(filepath=model_path,
                                monitor='val_accuracy',
                                save_weights_only=True,     #weights들만 저장하기!
                                verbose=1,
                                save_best_only=True)

## early stopping
cb_early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)


## 저장된 모델 지우기

# get_ipython().system("rm '/content/drive/MyDrive/Models/'*")


# ## 3) 모델 학습

## 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## 모델 학습
batch_size = 50

hist = model.fit(padded
                , labels_train.reshape((-1,1))
                , validation_data=(valid_padded, valid_labels.reshape((-1,1))),
                 epochs=100, batch_size=batch_size, callbacks=[cb_checkpoint, cb_early_stopping])

# conv_blocks = []
# for size in filter_sizes:
#     conv = keras.layers.Conv1D(filters=num_filters, kernel_size=size, padding="valid",
#                          activation="relu", strides=1)(z)
#     conv = keras.layers.MaxPooling1D(pool_size=2)(conv)
#     conv = keras.layers.Flatten()(conv)
#     conv_blocks.append(conv)

# z = keras.layers.Concatenate()(conv_blocks)
# z = keras.layers.Dropout(dropout)(z)
# z = keras.layers.Dense(hidden_dims, activation="relu")(z)
# model_output = keras.layers.Dense(1, activation="sigmoid")(z)

# model = keras.Model(model_input, model_output)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# vocab_size = 52170
# embed_dim = 200
# max_length = 30
# trainable = False
# dropout = 0.5
# input_size = (max_length, )     #train_sequences
# inputs = Input(shape=input_size)


# embed = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=max_length, trainable=trainable)(inputs)

# filter_sizes = [3, 4, 5]
# conv_blocks = []
# for size in filter_sizes:
#   conv = Conv1(filters=64, kernel_size=size, activation='relu' , padding='same')(embed)
#   pool = MaxPool1D()(conv)
#   conv_blokcs.append(pool)

# concat_conv = Concatenate()(conv_blocks)

# flat = Flatten()(concat_conv)

# dense = Dense(64, ativation='relu')(flat)
# drop = Dropout(dropout)(dense)
# outputs = Dense(1, activation='softmax')(drop)



# ## 모델 생성

# model(input = inputs, output=outputs)
# model.summary()


# # ## 2) Model 저장, Callback 설정

# from keras.callbacks import ModelCheckpoint, EarlyStopping

# ## 모델 weights 저장
# MODEL_SAVE_FOLDER = '/content/drive/MyDrive/2022mulcam/models/'
# model_path = f"{MODEL_SAVE_FOLDER}CNN-{{epoch:d}}-{{val_loss:.5f}}-{{val_accuracy:.5f}}.hdf5"

# ## checkpoint 설정
# cb_checkpoint = ModelCheckpoint(filepath=model_path,
#                                 monitor='val_accuracy',
#                                 save_weights_only=True,     #weights들만 저장하기!
#                                 verbose=1,
#                                 save_best_only=True)

# ## early stopping
# cb_early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)


# ## 저장된 모델 지우기

# # get_ipython().system("rm '/content/drive/MyDrive/Models/'*")


# # ## 3) 모델 학습

# ## 모델 컴파일
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ## 모델 학습
# batch_size = 50

# hist = model.fit(padded, labels_train, validation_data=(valid_padded, valid_labels),
#                  epochs=100, batch_size=batch_size, callbacks=[cb_checkpoint, cb_early_stopping])