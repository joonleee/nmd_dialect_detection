# 회귀 모델 학습 코드

# 사용 패키지
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import minmax_scale 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 

# 데이터셋 로드
X = np.load('MFCC_feat_all_20.npy')
X.shape 
y = np.load('MFCC_label_all_20.npy')
y.shape 

# X, y변수 전처리 - x변수 : 정규화 / y변수 : one-hot 인코딩 
x_data = minmax_scale(X) 
y_data = OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()
y_data

# 훈련용, 테스트용 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.3)

# X, Y변수 type 일치 - float32
X = tf.constant(x_train, tf.float32) 
y = tf.constant(y_train, tf.float32) 
X_val = tf.constant(x_val, tf.float32)
Y_val = tf.constant(y_val, tf.float32)

# w, b변수 정의
w = tf.Variable(tf.random.normal(shape=[20, 5])) 
b = tf.Variable(tf.random.normal(shape=[5])) 

# 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b 
    return model 

# softmax 함수
def soft_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.softmax(model) 
    return y_pred 

# 손실함수
def loss_fn() : # 인수 없음 
    y_pred = soft_fn(X)
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss

# 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.1)

# 반복학습 
for step in range(500) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
  
# 최적화된 model 검증 
y2_pred = soft_fn(X_val).numpy() 
y2_pred = tf.argmax(y2_pred, axis = 1) 
y2_true = tf.argmax(Y_val, axis = 1) 
acc = accuracy_score(y2_true, y2_pred)
print('accuracy =',acc) 






