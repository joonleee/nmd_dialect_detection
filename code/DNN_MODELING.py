# DNN 학습 모델 및 하나의 오디오 셋을 학습모델에 적용하여 예측하는 코드

# 사용 패키지
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import minmax_scale 
from sklearn.metrics import accuracy_score   
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 
import time
import librosa

# 데이터셋 로드 및 형태 확인
X = np.load('MFCC_feat_all_20.npy')
X.shape 
y = np.load('MFCC_label_all_20.npy')
y.shape 

# 데이터 전처리 : X변수 정규화
X = minmax_scale(X) 

# y변수 : label encoding -> one hot encoding
y = to_categorical(y) 
y.shape

# 훈련용, 테스트용 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# keras layer & model 생성
model = Sequential()
model.add(Dense(units=256, input_shape =(20, ), activation = 'relu')) 
model.add(Dense(units=128, activation = 'relu'))
model.add(Dense(units=64, activation = 'relu')) 
model.add(Dense(units=32, activation = 'relu')) 
model.add(Dense(units=5, activation = 'softmax'))
model.summary()

# model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='Adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

# model training(콜백 설정, 학습 소요 시간 확인 코드 포함)
callback = EarlyStopping(monitor='val_loss', patience=5)
start_time = time.time()
model_fit = model.fit(x=x_train, y=y_train, 
          epochs=50, 
          batch_size = 64, 
          verbose=1, 
          validation_data=(x_test, y_test), 
          callbacks=[callback]) 
stop_time = time.time() - start_time

# model evaluation 
model.evaluate(x_test, y_test)
x_train, x_test, y_train, y_test = train_test_split(
     X, y, test_size=0.5)
y_pred = model.predict(x_test)
y_pred = tf.argmax(input=y_pred, axis=1)
y_true = tf.argmax(input=y_test, axis=1)
acc = accuracy_score(y_true, y_pred)
print('accuracy=',acc)


# validation
X2 = np.load('val20_MFCC_feat.npy')
y2 = np.load('val20_MFCC_label.npy')
X2 = minmax_scale(X2) 
y2 = to_categorical(y2) 
y2.shape 
y_pred = model.predict(X2)
y_pred.shape 
y_pred = tf.argmax(input=y_pred, axis=1)
y_true = tf.argmax(input=y2, axis=1)
acc = accuracy_score(y_true, y_pred)
print('accuracy=',acc)

# model save
# model.save('학습모델_DNN.h5')

# 하나의 오디오 파일을 적용시켜 학습 모델 확인
audio, sr = librosa.load('jejugrandma.wav')
mf = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
m_test = np.mean(mf.T,axis=0)
M_test = minmax_scale(m_test) 
M_test = M_test.reshape(1,-1)
m_pred = model.predict(M_test)
m_pred = tf.argmax(input=m_pred, axis=1)
m_pred


