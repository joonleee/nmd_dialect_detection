# JSON파일과 WAV파일을 비교하여 1.5초이상의 WAV파일만 선별하고 해당 WAV 파일을 chunk로 잘라 저장하는 코드

# 사용 패키지
import json
import glob
import pandas as pd
import os
from pydub import AudioSegment
from tqdm.auto import tqdm

# 변수 선언 및 확장자 제거
jsonPath = 'E:/DATA/DC/DC_label'
jsontmp = glob.glob1(jsonPath,'*.json')
jsonList = [jsontmp[i][:-5] for i in range(len(jsontmp))]

wavPath = 'C:/Users/DBro/Desktop/finalproject/DC/DC_wav'
wavtmp = glob.glob1(wavPath,'*.wav')
wavList = [wavtmp[i][:-4] for i in range(len(wavtmp))]

bothList = [value for value in jsonList if value in wavList]

# 청크 & 에러 카운트 확인
chunkCnt = 0
errorCnt = 0
for file in tqdm(bothList):
    with open(r'E:/DATA/DC/DC_label/{0}.json'.format(file), 'r', encoding='UTF-8') as voice_json:
        jsonfile = json.load(voice_json)
        
    for line in jsonfile['utterance']:
        if line['end'] - line['start'] >= 1.5:
            chunkCnt += 1
        else:
            errorCnt += 1
            
print('청크:', chunkCnt)
print('에러:', errorCnt)
print('총:', chunkCnt+errorCnt)

# JSON의 START, END을 가져와서 1.5 청크로 자르는 반복문
for file in bothList:
    with open(r'E:/DATA/DC/DC_label/{0}.json'.format(file), 'r', encoding='UTF-8') as voice_json:
        jsonfile = json.load(voice_json)

    infoText = []
    infoStart = []
    infoEnd = []
    for line in jsonfile['utterance']:
        if line['end'] - line['start'] >= 1.5:
            infoText.append(line['standard_form'])
            infoStart.append(line['start'])
            infoEnd.append(line['end'])
    jsonDF = pd.DataFrame({'text': infoText,'start': infoStart,'end': infoEnd})

    audio_file= 'C:/Users/DBro/Desktop/finalproject/DC/DC_wav/{0}.wav'.format(file)
    audio = AudioSegment.from_wav(audio_file)

    for idx,t in tqdm(enumerate(infoStart)):
        if idx == len(infoStart):
            break
        audio_chunk=audio[infoStart[idx]*1000 : infoEnd[idx]*1000]
        audio_chunk.export('C:/Users/DBro/Desktop/finalproject/DC/DC_wav/{}_{}.wav'.format(file, idx), format='wav')

    os.remove('C:/Users/DBro/Desktop/finalproject/DC/DC_wav/{}.wav'.format(file))

