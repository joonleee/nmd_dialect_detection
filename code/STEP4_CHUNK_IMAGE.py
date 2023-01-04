# 5초 청크를 MEL SPECTROGRAM 이미지로 변환하여 저장하는 코드

# 사용 패키지
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
from tqdm.auto import tqdm
import random

# 변수 선언 및 확장자 제거
c5Path = 'C:/Users/DBro/Desktop/finalproject/DC/DC_c5/'
c5tmp = glob.glob1(c5Path,'*.wav')
c5List = [c5tmp[i][:-4] for i in range(len(c5tmp))]

c5List_rand = random.sample(c5List, 100000)

# 5초 청크를 MEL SPECTROGRAM 이미지로 변환하여 저장하는 반복문 

matplotlib.use('Agg')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for file in tqdm(c5List_rand):        
        y, sr = librosa.load('C:/Users/DBro/Desktop/finalproject/DC/DC_c5/{}.wav'.format(file))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr = sr, fmax = 8000, ax = ax)
        
        plt.savefig('E:/DATA/DC/DC_img/{}.png'.format(file), dpi = 10, bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)