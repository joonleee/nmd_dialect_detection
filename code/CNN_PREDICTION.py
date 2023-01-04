# CNN model을 불러와서 예측하는 코드

import tensorflow as tf

# 변수 선언
model = tf.keras.models.load_model('./model/CNN_conv3_dense3_epoch20.h5')
prediction = model.predict(X_test)

# 해당 CNN모델로 예측하는 반복문
for i in prediction:
    pre_ans = i.argmax()
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = '충청도'
    elif pre_ans == 1: pre_ans_str = '강원도'
    elif pre_ans == 2: pre_ans_str = '전라도'
    elif pre_ans == 3: pre_ans_str = '경상도'
    elif pre_ans == 4: pre_ans_str = '제주도'

    if i[0] >= 0.7: 
        print('해당 이미지는 '+pre_ans_str+'으로 추정됩니다.')
    if i[1] >= 0.7: 
        print('해당 이미지는 '+pre_ans_str+'로 추정됩니다.')
    if i[2] >= 0.7: 
        print('해당 이미지는 '+pre_ans_str+'로 추정됩니다.')
    if i[3] >= 0.7: 
        print('해당 이미지는 '+pre_ans_str+'로 추정됩니다.')
    if i[4] >= 0.7: 
        print('해당 이미지는 '+pre_ans_str+'로 추정됩니다.')