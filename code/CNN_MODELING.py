#CNN 모델 구축하는 코드

#사용 패키지
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

# 변수 선언
X = np.load('C:/DATA/img/img_array_all.npy')
y = np.load('C:/DATA/img/img_label_all.npy')

y = np_utils.to_categorical(y)
callback = EarlyStopping(monitor='val_loss', patience=5)

# RGB 스케일링
X = X.astype(float) / 255.0

# TRAIN/TEST Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# 모델 구조
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(37, 50, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(5, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics = ['accuracy'])

# 모델 학습
history = model.fit(x=x_train,y=y_train,
                    epochs=20,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    callbacks=[callback])

# 모델 평가
model.evaluate(x=x_test, y=y_test)

# 모델 저장
# dir = 'C:/DATA/img/CNN_conv3_dense3_epoch20.h5'
model.save(dir)