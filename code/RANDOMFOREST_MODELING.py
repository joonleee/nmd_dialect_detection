# RandomForest 학습 모델 및 하나의 오디오 셋을 학습모델에 적용하여 예측하는 코드

# 사용 패키지
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import joblib
import librosa

# 데이터셋 로드 및 형태 확인
X = np.load('MFCC_features_all.npy')
X.shape # (2611202, 40)

y = np.load('MFCC_label_all.npy')
y.shape # (2611202,)

# 랜덤포레스트 모델 model 생성 
obj = RandomForestClassifier() 
model = obj.fit(X, y)

# 검증셋 로드 및 데이터 처리  
x_test = np.load('mfcc_feature.npy')
y_test = np.load('mfcc_target.npy')
y_pred = model.predict(X = x_test) 
y_pred.replace("'","")
y_pred = y_pred.astype('int64')

# model 평가 
acc = accuracy_score(y_pred, y_test)
print(acc) 

# k겹 교차검정(k=5)
score = cross_validate(model, X, y, cv=5)
print(score)

# 모델 저장
# joblib.dump(model,'학습모델_RandomForest.joblib')

# 하나의 오디오 파일을 적용시켜 학습 모델 확인
audio, sr = librosa.load('jejugrandma.wav')
mf = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
m_test = np.mean(mf.T,axis=0)
M_test = m_test.reshape(1,-1)
y_pred = model.predict(X = M_test) 
y_pred