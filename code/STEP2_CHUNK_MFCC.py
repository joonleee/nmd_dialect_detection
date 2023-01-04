# JSON기준으로 잘려진 1.5초이상 청크를 MFCC FEATURES로 변환하여 저장하는 코드

# 사용 패키지
import glob
import numpy as np
import librosa
import warnings
from tqdm.auto import tqdm

# 변수 선언
region = ['DC', 'DG', 'DJ', 'DK', 'DZ']
PATH = 'D:/DATA/'

mfccs = []
mfcc_label = []

# 오디오 청크를 MFCC FEATURES로 변환하여 ARRAY로 저장하는 반복문
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for reg in region:
        wavtmp = glob.glob1(PATH+'/{}/{}_wav'.format(reg,reg),'*.wav')
        wavList = [wavtmp[i][:-4] for i in range(len(wavtmp))]
        wavList.sort()
         
        try:
            for file in tqdm(wavList):
                audio, sr = librosa.load('{}/{}/{}_wav/{}.wav'.format(PATH,reg,reg,file), res_type='kaiser_fast')
                feats = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
                mfccs_processed = np.mean(feats.T, axis=0).tolist()
                mfccs.append(mfccs_processed)
                mfcc_label.append('{}'.format(reg))
        except Exception as e:
            print('Some error at', file, e)
            pass

np.save('D:/DATA/NPY/MFCC_feature20', mfccs)
np.save('D:/DATA/NPY/MFCC_target20', mfcc_label)