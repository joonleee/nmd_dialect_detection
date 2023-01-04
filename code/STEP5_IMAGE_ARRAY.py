# MEL SPECTROGRAM 이미지를 ARRAY로 변환하고 저장하는 코드

# 사용 패키지
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
from numpy import asarray

# 변수 선언
PATH = 'C:/DATA/img'
region = ['DC','DG', 'DJ', 'DK', 'DZ']
  
DC_img = []
DG_img = []
DJ_img = []
DK_img = []
DZ_img = []

DC_label = []
DG_label = []
DJ_label = []
DK_label = []
DZ_label = []

DC_L = 0
DG_L = 1
DJ_L = 2
DK_L = 3
DZ_L = 4

# RGB format으로 이미지를 ARRAY로 변환하고 저장하는 반복문
for reg in region:
    imgtmp = glob.glob1(PATH+'/{}_img'.format(reg),'*.png')
    imgList = [imgtmp[i][:-4] for i in range(len(imgtmp))]
    imgList.sort()
   
    for file in tqdm(imgList):
        img_file= 'C:/DATA/img/{}_img/{}.png'.format(reg, file)
        image = Image.open(img_file).convert('RGB')
        numpyimage = asarray(image)
        if file.startswith('DC'):
            DC_img.append(numpyimage)
            DC_label.append(DC_L)
        elif file.startswith('DG'):
            DG_img.append(numpyimage)
            DG_label.append(DG_L)
        elif file.startswith('DJ'):
            DJ_img.append(numpyimage)
            DJ_label.append(DJ_L)
        elif file.startswith('DK'):
            DK_img.append(numpyimage)
            DK_label.append(DK_L)
        else: 
            DZ_img.append(numpyimage)
            DZ_label.append(DZ_L)

img_array_all = np.concatenate((DC_img, DG_img, DJ_img, DK_img, DZ_img), axis = 0)
img_label_all = np.concatenate((DC_label, DG_label, DJ_label, DK_label, DZ_label), axis = 0)