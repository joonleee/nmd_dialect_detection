# 기존에 있던 청크를 5초단위로 자르고 저장하는 코드

# 사용 패키지
import glob
from pydub import AudioSegment
from tqdm.auto import tqdm
import contextlib
import wave

# 변수 선언 및 확장자 제거

chunkPath = 'C:/Users/DBro/Desktop/finalproject/DC/DC_wav'
chunktmp = glob.glob1(chunkPath,'*.wav')
chunkList = [chunktmp[i][:-4] for i in range(len(chunktmp))]


# 기존에 있던 청크를 5초 단위로 자르고 저장하는 반복문 

for file in tqdm(chunkList):
    audio_file= 'C:/Users/DBro/Desktop/finalproject/DC/DC_wav/{0}.wav'.format(file)
    audio = AudioSegment.from_wav(audio_file)
    with contextlib.closing(wave.open(audio_file,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    if (duration / 5 >= 3):
        audio_chunk=audio[0:5000]
        audio_chunk1=audio[5000:10000]
        audio_chunk2=audio[10000:15000]
        audio_chunk.export('C:/Users/DBro/Desktop/finalproject/DC/DC_c5/{}_0.wav'.format(file),format='wav')
        audio_chunk1.export('C:/Users/DBro/Desktop/finalproject/DC/DC_c5//{}_1.wav'.format(file),format='wav')
        audio_chunk2.export('C:/Users/DBro/Desktop/finalproject/DC/DC_c5//{}_2.wav'.format(file),format='wav')

                        
    elif (duration / 5 >= 2 and duration / 5 < 3):
        audio_chunk=audio[0:5000]
        audio_chunk1=audio[5000:10000]
        audio_chunk.export('C:/Users/DBro/Desktop/finalproject/DC/DC_c5/{}_0.wav'.format(file),format='wav')
        audio_chunk1.export('C:/Users/DBro/Desktop/finalproject/DC/DC_c5/{}_1.wav'.format(file),format='wav')
        
    elif (duration / 5 >= 1 and duration / 5 < 2):
        audio_chunk=audio[0:5000]
        audio_chunk.export('C:/Users/DBro/Desktop/finalproject/DC/DC_c5/{}_0.wav'.format(file),format='wav')